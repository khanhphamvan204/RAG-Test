from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.types import Command
import operator
import logging
import json
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
from app.config import Config

from app.services.rag_service import rag_search_tool
from app.services.activity_search_service import (
    activity_search_with_llm_tool,
    set_bearer_token
)

load_dotenv()

logger = logging.getLogger(__name__)

MAX_MESSAGES = int(os.getenv('MAX_CONTEXT_MESSAGES', '10'))
CONTEXT_STRATEGY = os.getenv('CONTEXT_STRATEGY', 'keep_system')

_current_unit_name = "default_unit"

def set_current_unit_name(unit_name: str):
    """Set unit_name để RAG tool sử dụng"""
    global _current_unit_name
    _current_unit_name = unit_name
    logger.info(f"[LANGGRAPH] Set current unit_name: {unit_name}")

def get_current_unit_name() -> str:
    """Get unit_name hiện tại"""
    return _current_unit_name


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_role: str
    user_id: int
    unit_name: str
    # HITL fields
    needs_clarification: bool
    clarification_type: str  # "activity_filter", "date_range", etc.
    clarification_message: str  # Message để show cho user


def trim_messages(messages: list, max_messages: int = MAX_MESSAGES, strategy: str = CONTEXT_STRATEGY) -> list:
    """
    Quản lý context window với nhiều chiến lược
    ĐẶC BIỆT: Đảm bảo không phá vỡ chuỗi AIMessage-ToolMessage cho Gemini
    
    Strategies:
    - sliding_window: Giữ N messages gần nhất
    - keep_system: Giữ system message + N messages gần nhất
    - summarize: Tóm tắt messages cũ (cần thêm LLM call)
    """
    if len(messages) <= max_messages:
        return messages
    
    logger.info(f"Trimming {len(messages)} messages to {max_messages} using strategy: {strategy}")
    
    if strategy == 'sliding_window':
        trimmed = messages[-max_messages:]
        # Đảm bảo không bắt đầu bằng ToolMessage mồ côi
        if trimmed and isinstance(trimmed[0], ToolMessage):
            # Tìm AIMessage trước đó có tool_calls
            for i in range(len(messages) - max_messages - 1, -1, -1):
                if isinstance(messages[i], AIMessage) and hasattr(messages[i], 'tool_calls') and messages[i].tool_calls:
                    # Loại bỏ ToolMessage mồ côi
                    trimmed = [msg for msg in trimmed if not isinstance(msg, ToolMessage) or msg != trimmed[0]]
                    break
        return trimmed
    
    elif strategy == 'keep_system':
        # Luôn giữ SystemMessage đầu tiên (CHỈ MỘT)
        system_message = None
        other_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage) and system_message is None:
                system_message = msg
            elif not isinstance(msg, SystemMessage):
                other_messages.append(msg)
        
        if len(other_messages) > max_messages - (1 if system_message else 0):
            # Tính toán số lượng messages cần giữ
            keep_count = max_messages - (1 if system_message else 0)
            candidate_messages = other_messages[-keep_count:]
            
            # Kiểm tra nếu message đầu tiên là ToolMessage mồ côi
            while candidate_messages and isinstance(candidate_messages[0], ToolMessage):
                # Tìm xem có AIMessage với tool_calls tương ứng không
                found_pair = False
                tool_msg = candidate_messages[0]
                
                # Tìm ngược trong other_messages
                start_index = len(other_messages) - keep_count - 1
                for i in range(start_index, -1, -1):
                    if (isinstance(other_messages[i], AIMessage) and 
                        hasattr(other_messages[i], 'tool_calls') and 
                        other_messages[i].tool_calls):
                        found_pair = True
                        break
                
                if not found_pair:
                    # Loại bỏ ToolMessage mồ côi
                    candidate_messages = candidate_messages[1:]
                    if not candidate_messages:
                        break
                else:
                    # Có cặp, nhưng không nằm trong candidate, loại bỏ ToolMessage
                    candidate_messages = candidate_messages[1:]
                    if not candidate_messages:
                        break
            
            other_messages = candidate_messages
        
        # Trả về: SystemMessage (nếu có) + other_messages
        result = []
        if system_message:
            result.append(system_message)
        result.extend(other_messages)
        return result
    
    elif strategy == 'keep_first_last':
        first_n = max_messages // 2
        last_n = max_messages - first_n
        result = messages[:first_n] + messages[-last_n:]
        
        # Kiểm tra và sửa ToolMessage mồ côi ở giữa
        if len(result) > first_n and isinstance(result[first_n], ToolMessage):
            result = result[:first_n] + [msg for msg in result[first_n:] if not isinstance(msg, ToolMessage) or msg != result[first_n]]
        
        return result
    
    else:
        return messages[-max_messages:]


def create_agent_node(llm_with_tools):
    def agent(state: AgentState):
        messages = state["messages"]
        user_role = state.get("user_role", "student")
        user_id = state.get("user_id", 0)
        unit_name = state.get("unit_name", "default_unit")
        
        # ========= HITL: Không cần check clarification_message nữa =========
        # Clarification_checker đã handle bằng Command(goto=END)
        # Agent chỉ chạy khi không có clarification hoặc user đã response
        
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime("%d/%m/%Y")
        current_time_str = current_datetime.strftime("%H:%M:%S")
        current_weekday = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][current_datetime.weekday()]
        
        system_context = f"""
Bạn là trợ lý AI thông minh, hỗ trợ người dùng trong hệ thống quản lý học tập.

VAI TRÒ CỦA BẠN:
- Bạn là công cụ hỗ trợ, KHÔNG phải là cố vấn hay sinh viên
- Người dùng đang tương tác với bạn có vai trò: {user_role} (advisor = cố vấn, student = sinh viên)
- Bạn giúp người dùng tìm kiếm thông tin, tra cứu tài liệu, và truy vấn hoạt động

THÔNG TIN THỜI GIAN HIỆN TẠI:
- Ngày hiện tại: {current_weekday}, {current_date_str}
- Giờ hiện tại: {current_time_str}

THÔNG TIN NGƯỜI DÙNG ĐANG HỎI:
- Vai trò: {user_role} {'(Cố vấn học tập)' if user_role == 'advisor' else '(Sinh viên)' if user_role == 'student' else ''}
- ID: {user_id}
- Đơn vị: {unit_name}

CÔNG CỤ TÌM KIẾM:
1. vector_rag_search - Tìm kiếm tài liệu, quy định trong thư viện của "{unit_name}"
2. activity_search_with_summary - Tìm kiếm hoạt động ngoại khóa, sự kiện

HƯỚNG DẪN TRẢ LỜI:

KHI CHÀO HỎI hoặc CHAT THÔNG THƯỜNG:
- TRẢ LỜI TRỰC TIẾP một cách thân thiện, tự nhiên
- KHÔNG gọi tool, KHÔNG nhắc đến vai trò của user
- Ví dụ đúng: "Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?"
- Ví dụ SAI: "Chào bạn, tôi có thể giúp gì với vai trò cố vấn..."

CHỈ GỌI TOOL khi user hỏi cụ thể về:
- Tài liệu, quy định, quy trình → dùng vector_rag_search
- Hoạt động, sự kiện → dùng activity_search_with_summary

KIẾN THỨC QUAN TRỌNG:
Hoạt động có 2 loại điểm: CTXH (hiến máu, tình nguyện) và Rèn luyện (workshop, cuộc thi)

vector_rag_search: Dùng khi user hỏi về:
- Quy định, quy trình, nội quy
- Tài liệu hướng dẫn
- Thông tin chung về hệ thống
- Bất kỳ câu hỏi nào liên quan đến TÀI LIỆU của đơn vị

QUAN TRỌNG khi gọi vector_rag_search:
- Tool sẽ TỰ ĐỘNG search trong thư viện của đơn vị "{unit_name}"
- KHÔNG CẦN truyền unit_name vào tool call
- Ví dụ đúng: vector_rag_search(query="quy định học vụ", k=5)
- Tool sẽ tự động lấy unit_name từ context

activity_search_with_summary: Dùng khi user hỏi về hoạt động ngoại khóa
- Ví dụ: "Hoạt động sắp tới?", "Hoạt động CTXH", "Workshop", "Điểm rèn luyện"

CÁCH GỌI TOOL ACTIVITY:
activity_search_with_summary(
    user_role="{user_role}",
    user_id={user_id},
    status="upcoming"
)

LƯU Ý QUAN TRỌNG:
- KHÔNG BAO GIỜ truyền bearer_token vào tool call (hệ thống tự động xử lý)
- KHÔNG BAO GIỜ truyền unit_name vào vector_rag_search (hệ thống tự động lấy từ context)
- Với hoạt động, LUÔN dùng activity_search_with_summary (có tóm tắt LLM)
- Nếu tool trả về total=0, DỪNG và trả lời "Không có hoạt động/tài liệu phù hợp"
- KHÔNG suy đoán hoặc tự tạo dữ liệu hoạt động/tài liệu

KHI USER HỎI VỀ LỌC HOẠT ĐỘNG:
- Phân tích yêu cầu của user từ ngôn ngữ tự nhiên
- Extract filters: điểm rèn luyện, thời gian, khoa, trạng thái
- Gọi activity_search_with_summary với parameters phù hợp

KHI CẦN CLARIFICATION:
- Nếu user hỏi quá chung (VD: "Hoạt động nào?", "Activities?") mà bạn KHÔNG thể xác định được filters → HỎI lại user
- Ví dụ trả lời: "Bạn muốn lọc theo tiêu chí nào? (điểm rèn luyện, thời gian, khoa, ...)"
- Nếu user đã nói rõ tiêu chí (VD: "Hoạt động kiếm điểm CTXH", "Hoạt động tháng này") → KHÔNG hỏi lại, gọi tool ngay
"""
        
        # GEMINI FIX: Loại bỏ tất cả SystemMessage cũ, chỉ giữ SystemMessage mới ở đầu
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Xây dựng messages mới: SystemMessage + non_system_messages
        full_messages = [SystemMessage(content=system_context)] + non_system_messages
        
        # Trim messages (giữ SystemMessage đầu tiên)
        full_messages = trim_messages(full_messages)
        
        provider_name = Config.LLM_PROVIDER.upper()
        logger.info(f"[AGENT] Sending {len(full_messages)} messages to {provider_name} (unit: {unit_name}):")
        for idx, msg in enumerate(full_messages):
            msg_type = type(msg).__name__
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            logger.info(f"  [{idx}] {msg_type}{' (with tool_calls)' if has_tool_calls else ''}")
        
        response = llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    return agent





def should_continue(state: AgentState):
    """Routing sau khi agent chạy"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return END


def create_llm(provider: str = None, temperature: float = 0.3):
    """
    Factory function to create LLM based on provider
    
    Args:
        provider: "gemini" or "openai" (defaults to Config.LLM_PROVIDER)
        temperature: Model temperature setting
    
    Returns:
        ChatModel instance (Gemini or OpenAI)
    
    Raises:
        ValueError: If provider is invalid or API key is missing
    """
    if provider is None:
        provider = Config.LLM_PROVIDER.lower()
    
    logger.info(f"[LLM_FACTORY] Creating LLM with provider: {provider}")
    
    if provider == "gemini":
        api_key = Config.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables")
        
        model_name = Config.GEMINI_MODEL
        logger.info(f"[LLM_FACTORY] Using Gemini model: {model_name}")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
    
    elif provider == "openai":
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        model_name = Config.OPENAI_MODEL
        logger.info(f"[LLM_FACTORY] Using OpenAI model: {model_name}")
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature
        )
    
    elif provider == 'groq':
        api_key = Config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        model_name = Config.GROQ_MODEL
        logger.info(f"[LLM_FACTORY] Using Groq model: {model_name}")
        
        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {provider}. Must be 'gemini', 'openai', or 'groq'")


def create_langgraph():
    # Create LLM using factory
    llm = create_llm(temperature=0.3)
    
    tools = [
        rag_search_tool,
        activity_search_with_llm_tool
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    workflow = StateGraph(AgentState)
    
    # Simplified workflow: agent tự quyết định khi nào cần clarification
    workflow.add_node("agent", create_agent_node(llm_with_tools))
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point trực tiếp tới agent
    workflow.set_entry_point("agent")
    
    # Agent routing: gọi tools hoặc kết thúc
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    
    # Tools -> agent
    workflow.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    model_name = Config.GEMINI_MODEL if Config.LLM_PROVIDER == 'gemini' else (Config.OPENAI_MODEL if Config.LLM_PROVIDER == 'openai' else Config.GROQ_MODEL)
    logger.info(f"LangGraph: Using {Config.LLM_PROVIDER} ({model_name}) with MemorySaver, context strategy: {CONTEXT_STRATEGY}, max messages: {MAX_MESSAGES}")

    return workflow.compile(checkpointer=checkpointer)


graph = create_langgraph()


def process_query(
    query: str,
    user_role: str = "student",
    user_id: int = 0,
    bearer_token: str = None,
    unit_name: str = "default_unit",
    thread_id: str | None = None
) -> str:
    """
    Process user query với LangGraph agent
    
    Args:
        query: Câu hỏi của user
        user_role: Role của user (student/advisor/admin)
        user_id: ID của user
        bearer_token: JWT token để call external APIs
        unit_name: Tên đơn vị của user (VD: "Khoa Công nghệ Thông tin")
        thread_id: ID của conversation thread (optional)
    
    Returns:
        JSON string với response
    """
    try:
        # Set bearer token cho activity search
        if bearer_token:
            set_bearer_token(bearer_token)
            logger.info(f"[PROCESS_QUERY] Token set: {bearer_token[:20]}...")
        else:
            logger.warning("[PROCESS_QUERY] No bearer token provided")
        
        # ========= Set unit_name cho RAG search =========
        set_current_unit_name(unit_name)
        logger.info(f"[PROCESS_QUERY] Unit set: {unit_name}")
        
        # Generate thread_id nếu chưa có
        if not thread_id:
            thread_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
            logger.info(f"[PROCESS_QUERY] Generated thread_id: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check conversation history
        state = graph.get_state(config)
        current_messages = state.values.get("messages", []) if state else []
        logger.info(f"[PROCESS_QUERY] Thread {thread_id} has {len(current_messages)} messages in history")
        
        # ========= Prepare initial state =========
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_role": user_role,
            "user_id": user_id,
            "unit_name": unit_name,
            # Initialize HITL fields
            "needs_clarification": False,
            "clarification_type": "",
            "clarification_message": ""
        }
        
        logger.info(f"[PROCESS_QUERY] Processing query from user {user_id} ({user_role}) in unit '{unit_name}'")
        
        # Invoke graph
        result = graph.invoke(initial_state, config)
        
        # Lấy response message từ result
        messages = result.get("messages", [])
        if not messages:
            return json.dumps({
                "status": "error",
                "data": None,
                "error": "Không có phản hồi",
                "thread_id": thread_id
            }, ensure_ascii=False, indent=2)
        
        last_message = messages[-1]
        response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Extract activities và source từ tool messages
        activities_raw = []
        source = "general"
        total_activities = 0
        
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        
        if tool_messages:
            last_tool_msg = tool_messages[-1]
            try:
                # Kiểm tra content có rỗng hoặc không hợp lệ
                if not last_tool_msg.content or (isinstance(last_tool_msg.content, str) and last_tool_msg.content.strip() == ""):
                    logger.warning(f"[PROCESS_QUERY] Tool returned empty content")
                else:
                    tool_result = json.loads(last_tool_msg.content) if isinstance(last_tool_msg.content, str) else last_tool_msg.content
                    
                    if isinstance(tool_result, dict):
                        if tool_result.get('source') == 'activity':
                            activities_raw = tool_result.get('activities_raw', [])
                            total_activities = tool_result.get('total', 0)
                            source = 'activity'
                            logger.info(f"[PROCESS_QUERY] Found {total_activities} activities from tool")
                        elif tool_result.get('source') == 'rag':
                            source = 'rag'
                            logger.info(f"[PROCESS_QUERY] RAG search performed in unit '{unit_name}'")
            except json.JSONDecodeError as e:
                logger.error(f"[PROCESS_QUERY] Tool message JSON parse error: {e}. Content: {last_tool_msg.content[:100] if last_tool_msg.content else 'None'}")
            except Exception as e:
                logger.error(f"[PROCESS_QUERY] Tool message parse error: {e}")
        
        total_messages_after = len(messages)
        logger.info(f"[PROCESS_QUERY] Thread {thread_id} now has {total_messages_after} messages after processing")
        
        return json.dumps({
            "status": "success",
            "data": {
                "response": response_text,
                "user_role": user_role,
                "user_id": user_id,
                "unit_name": unit_name,
                "source": source,
                "activities": activities_raw,
                "total_activities": total_activities,
                "context_info": {
                    "total_messages": total_messages_after,
                    "max_messages": MAX_MESSAGES,
                    "strategy": CONTEXT_STRATEGY
                }
            },
            "error": None,
            "thread_id": thread_id
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"[PROCESS_QUERY] Process error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return json.dumps({
            "status": "error",
            "data": None,
            "error": str(e),
            "thread_id": thread_id
        }, ensure_ascii=False, indent=2)


def get_conversation_history(thread_id: str) -> list:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        
        logger.info(f"Retrieved {len(messages)} messages for thread: {thread_id}")
        return messages
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return []


def clear_conversation_history(thread_id: str) -> bool:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        state = graph.get_state(config)
        if state and state.values.get("messages"):
            logger.info(f"Cannot clear MemorySaver history for thread: {thread_id}")
            logger.info("MemorySaver will reset on server restart")
            return False
        
        return False
            
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return False


def get_context_stats(thread_id: str) -> dict:
    """
    Lấy thống kê về context window
    """
    try:
        messages = get_conversation_history(thread_id)
        
        human_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        ai_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
        tool_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))
        
        return {
            "thread_id": thread_id,
            "total_messages": len(messages),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "tool_messages": tool_count,
            "max_messages": MAX_MESSAGES,
            "strategy": CONTEXT_STRATEGY,
            "will_trim": len(messages) > MAX_MESSAGES
        }
    except Exception as e:
        logger.error(f"Get context stats error: {e}")
        return {}
