from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
import logging
import json
from dotenv import load_dotenv
import os

from app.services.rag_service import rag_search_tool
from app.services.activity_search_service import (
    activity_search_tool,
    activity_search_with_llm_tool
)

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_role: str
    user_id: int
    bearer_token: str


def create_agent_node(llm_with_tools):
    def agent(state: AgentState):
        messages = state["messages"]
        user_role = state.get("user_role", "student")
        user_id = state.get("user_id", 0)
        bearer_token = state.get("bearer_token", "")
        
        system_context = f"""
Bạn là trợ lý AI cho hệ thống quản lý cố vấn học tập.

Người dùng hiện tại:
- Vai trò: {user_role}
- ID: {user_id}
- Bearer token: ĐÃ ĐƯỢC CUNG CẤP

Công cụ có sẵn:
1. vector_rag_search - Tìm tài liệu
2. activity_search - Tìm hoạt động (dữ liệu thô)
3. activity_search_with_summary - Tìm hoạt động + tóm tắt LLM

QUAN TRỌNG - XÁC THỰC:
Khi gọi activity_search hoặc activity_search_with_summary, BẮT BUỘC phải truyền:
- user_role: "{user_role}"
- user_id: {user_id}
- bearer_token: "{bearer_token}"

QUAN TRỌNG - XỬ LÝ KẾT QUẢ RỖNG:
- Nếu tool trả về total=0 hoặc activities_raw=[], KHÔNG GỌI LẠI TOOL
- Chỉ gọi MỘT TOOL activity duy nhất cho mỗi câu hỏi
- Nếu không tìm thấy hoạt động, trả lời: "Hiện tại không có hoạt động nào phù hợp"
- KHÔNG suy đoán hoặc hallucinate dữ liệu hoạt động

Ví dụ:
activity_search_with_summary(user_role="{user_role}", user_id={user_id}, bearer_token="{bearer_token}", status="upcoming")

NẾU TOOL TRẢ VỀ 0 KẾT QUẢ:
→ DỪNG GỌI THÊM TOOL
→ Trả lời trực tiếp: "Hiện tại không có hoạt động nào phù hợp với yêu cầu của bạn"
"""
        
        full_messages = [SystemMessage(content=system_context)] + messages
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
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


graph = create_langgraph()


def process_query(
    query: str,
    user_role: str = "student",
    user_id: int = 0,
    bearer_token: str = None,
    thread_id: str | None = None
) -> str:
    try:
        if not bearer_token:
            logger.warning("[PROCESS] Không có bearer token")
        else:
            logger.info(f"[PROCESS] Token: {bearer_token[:30]}...")
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_role": user_role,
            "user_id": user_id,
            "bearer_token": bearer_token or ""
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
        
        # LOG TẤT CẢ MESSAGES ĐỂ DEBUG
        logger.info(f"[DEBUG] Tổng số messages: {len(messages)}")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            logger.info(f"[DEBUG] Message {i}: {msg_type}")
            if isinstance(msg, ToolMessage):
                logger.info(f"[DEBUG]   - Tool: {msg.name if hasattr(msg, 'name') else 'unknown'}")
                logger.info(f"[DEBUG]   - Nội dung xem trước: {str(msg.content)[:200]}")
        
        last_message = messages[-1]
        
        # TRÍCH XUẤT TEXT PHẢN HỒI
        response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # TRÍCH XUẤT KẾT QUẢ TOOL TỪ ToolMessage
        activities_raw = []
        source = "general"
        total_activities = 0
        
        # Tìm kiếm ToolMessage trong messages - LẤY TOOL MESSAGE CUỐI CÙNG
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        logger.info(f"[DEBUG] Tìm thấy {len(tool_messages)} tool messages")
        
        if tool_messages:
            # LẤY TOOL MESSAGE CUỐI CÙNG (mới nhất)
            last_tool_msg = tool_messages[-1]
            logger.info(f"[DEBUG] Sử dụng tool message cuối cùng: {last_tool_msg.name if hasattr(last_tool_msg, 'name') else 'unknown'}")
            
            try:
                tool_result = json.loads(last_tool_msg.content) if isinstance(last_tool_msg.content, str) else last_tool_msg.content
                
                logger.info(f"[EXTRACT] Kết quả tool: {tool_result}")
                
                # Kiểm tra nếu là activity tool
                if isinstance(tool_result, dict):
                    if tool_result.get('source') == 'activity':
                        activities_raw = tool_result.get('activities_raw', [])
                        total_activities = tool_result.get('total', 0)
                        source = 'activity'
                        logger.info(f"[EXTRACT] Tìm thấy {total_activities} hoạt động")
                    elif tool_result.get('source') == 'rag':
                        source = 'rag'
                        logger.info("[EXTRACT] Nguồn là RAG")
            except Exception as e:
                logger.error(f"[EXTRACT] Lỗi parse tool message: {e}")
        
        logger.info(f"[FINAL] source={source}, activities={len(activities_raw)}, total={total_activities}")
        
        return json.dumps({
            "status": "success",
            "data": {
                "response": response_text,
                "user_role": user_role,
                "user_id": user_id,
                "source": source,  # "rag", "activity", hoặc "general"
                "activities": activities_raw,  # Danh sách hoạt động raw (chỉ có nếu source="activity")
                "total_activities": total_activities  # Tổng số hoạt động
            },
            "error": None,
            "thread_id": thread_id
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"[PROCESS] Lỗi: {str(e)}")
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
        return state.values.get("messages", [])
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử hội thoại: {e}")
        return []


def clear_conversation_history(thread_id: str) -> bool:
    try:
        logger.warning("Xóa lịch sử hội thoại chưa được implement cho MemorySaver")
        return False
    except Exception as e:
        logger.error(f"Lỗi khi xóa lịch sử hội thoại: {e}")
        return False