from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
import operator
import logging
import json
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid

from app.services.rag_service import rag_search_tool
from app.services.activity_search_service import (
    activity_search_tool,
    activity_search_with_llm_tool,
    set_bearer_token
)

load_dotenv()

logger = logging.getLogger(__name__)

MAX_MESSAGES = int(os.getenv('MAX_CONTEXT_MESSAGES', '10'))
CONTEXT_STRATEGY = os.getenv('CONTEXT_STRATEGY', 'keep_system')


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_role: str
    user_id: int


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
        
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime("%d/%m/%Y")
        current_time_str = current_datetime.strftime("%H:%M:%S")
        current_weekday = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][current_datetime.weekday()]
        
        system_context = f"""
Bạn là trợ lý AI cho hệ thống quản lý cố vấn học tập.

THÔNG TIN THỜI GIAN HIỆN TẠI:
- Ngày hiện tại: {current_weekday}, {current_date_str}
- Giờ hiện tại: {current_time_str}

Người dùng hiện tại:
- Vai trò: {user_role}
- ID: {user_id}

Công cụ có sẵn:
1. vector_rag_search - Tìm kiếm tài liệu trong hệ thống
2. activity_search - Tìm kiếm hoạt động ngoại khóa (dữ liệu thô)
3. activity_search_with_summary - Tìm kiếm hoạt động + tóm tắt LLM

HƯỚNG DẪN SỬ DỤNG TOOLS:

vector_rag_search: Dùng khi user hỏi về:
- Quy định, quy trình, nội quy
- Tài liệu hướng dẫn
- Thông tin chung về hệ thống

activity_search: Dùng khi cần dữ liệu thô về hoạt động:
- Liệt kê tất cả hoạt động
- Export/báo cáo
- Xử lý dữ liệu phức tạp

activity_search_with_summary: Dùng khi user hỏi về hoạt động:
- "Có hoạt động gì sắp tới?"
- "Tìm hoạt động CTXH"
- "Hoạt động nào cho điểm rèn luyện?"

CÁCH GỌI TOOL ACTIVITY:
activity_search_with_summary(
    user_role="{user_role}",
    user_id={user_id},
    status="upcoming"
)

LƯU Ý QUAN TRỌNG:
- KHÔNG BAO GIỜ truyền bearer_token vào tool call (hệ thống tự động xử lý)
- Chỉ gọi MỘT TOOL activity duy nhất cho mỗi câu hỏi
- Nếu tool trả về total=0, DỪNG và trả lời "Không có hoạt động phù hợp"
- KHÔNG suy đoán hoặc tự tạo dữ liệu hoạt động
"""
        
        # GEMINI FIX: Loại bỏ tất cả SystemMessage cũ, chỉ giữ SystemMessage mới ở đầu
        # Gemini yêu cầu: SystemMessage chỉ ở đầu, sau đó là User/AI/Tool messages
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Xây dựng messages mới: SystemMessage + non_system_messages
        full_messages = [SystemMessage(content=system_context)] + non_system_messages
        
        # Trim messages (giữ SystemMessage đầu tiên)
        full_messages = trim_messages(full_messages)
        
        logger.info(f"Sending {len(full_messages)} messages to Gemini:")
        for idx, msg in enumerate(full_messages):
            msg_type = type(msg).__name__
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            logger.info(f"  [{idx}] {msg_type}{' (with tool_calls)' if has_tool_calls else ''}")
        
        response = llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    return agent


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return END


def create_langgraph():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.3
    )
    
    tools = [
        rag_search_tool,
        activity_search_tool,
        activity_search_with_llm_tool
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", create_agent_node(llm_with_tools))
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    logger.info(f"LangGraph: Using MemorySaver with context strategy: {CONTEXT_STRATEGY}, max messages: {MAX_MESSAGES}")

    return workflow.compile(checkpointer=checkpointer)


graph = create_langgraph()


def process_query(
    query: str,
    user_role: str = "student",
    user_id: int = 0,
    bearer_token: str = None,
    thread_id: str | None = None
) -> str:
    try:
        if bearer_token:
            set_bearer_token(bearer_token)
            logger.info(f"Token set: {bearer_token[:20]}...")
        else:
            logger.warning("No bearer token provided")
        
        if not thread_id:
            thread_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generated thread_id: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        state = graph.get_state(config)
        current_messages = state.values.get("messages", []) if state else []
        logger.info(f"Thread {thread_id} has {len(current_messages)} messages in history")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_role": user_role,
            "user_id": user_id
        }
        
        result = graph.invoke(initial_state, config)
        
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
        
        activities_raw = []
        source = "general"
        total_activities = 0
        
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        
        if tool_messages:
            last_tool_msg = tool_messages[-1]
            try:
                tool_result = json.loads(last_tool_msg.content) if isinstance(last_tool_msg.content, str) else last_tool_msg.content
                
                if isinstance(tool_result, dict):
                    if tool_result.get('source') == 'activity':
                        activities_raw = tool_result.get('activities_raw', [])
                        total_activities = tool_result.get('total', 0)
                        source = 'activity'
                    elif tool_result.get('source') == 'rag':
                        source = 'rag'
            except Exception as e:
                logger.error(f"Tool message parse error: {e}")
        
        total_messages_after = len(messages)
        logger.info(f"Thread {thread_id} now has {total_messages_after} messages after processing")
        
        return json.dumps({
            "status": "success",
            "data": {
                "response": response_text,
                "user_role": user_role,
                "user_id": user_id,
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
        logger.error(f"Process error: {str(e)}")
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