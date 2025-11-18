# ==================== FILE: app/services/langgraph_service.py ====================

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
from datetime import datetime

from app.services.rag_service import rag_search_tool
from app.services.activity_search_service import (
    activity_search_tool,
    activity_search_with_llm_tool,
    set_bearer_token  # ← IMPORT FUNCTION SET TOKEN
)

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_role: str
    user_id: int
    bearer_token: str  # Vẫn giữ trong state để tracking


def create_agent_node(llm_with_tools):
    def agent(state: AgentState):
        messages = state["messages"]
        user_role = state.get("user_role", "student")
        user_id = state.get("user_id", 0)
        
        # Lấy ngày giờ hiện tại
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime("%d/%m/%Y")
        current_time_str = current_datetime.strftime("%H:%M:%S")
        current_weekday = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][current_datetime.weekday()]
        
        # ĐÃ BỎ HOÀN TOÀN instruction về bearer_token
        system_context = f"""
Bạn là trợ lý AI cho hệ thống quản lý cố vấn học tập.

THÔNG TIN THỜI GIAN HIỆN TẠI:
- Ngày hiện tại: {current_weekday}, {current_date_str}
- Giờ hiện tại: {current_time_str}

Người dùng hiện tại:
- Vai trò: {user_role}
- ID: {user_id}

Công cụ có sẵn:
1. **vector_rag_search** - Tìm kiếm tài liệu trong hệ thống
2. **activity_search** - Tìm kiếm hoạt động ngoại khóa (dữ liệu thô)
3. **activity_search_with_summary** - Tìm kiếm hoạt động + tóm tắt LLM

HƯỚNG DẪN SỬ DỤNG TOOLS:

📚 **vector_rag_search**: Dùng khi user hỏi về:
- Quy định, quy trình, nội quy
- Tài liệu hướng dẫn
- Thông tin chung về hệ thống

🎯 **activity_search**: Dùng khi cần dữ liệu thô về hoạt động:
- Liệt kê tất cả hoạt động
- Export/báo cáo
- Xử lý dữ liệu phức tạp

✨ **activity_search_with_summary**: Dùng khi user hỏi về hoạt động:
- "Có hoạt động gì sắp tới?"
- "Tìm hoạt động CTXH"
- "Hoạt động nào cho điểm rèn luyện?"

CÁCH GỌI TOOL ACTIVITY:
activity_search_with_summary(
    user_role="{user_role}",
    user_id={user_id},
    status="upcoming"  # hoặc các filter khác
)

LƯU Ý QUAN TRỌNG:
- KHÔNG BAO GIỜ truyền bearer_token vào tool call (hệ thống tự động xử lý)
- Chỉ gọi MỘT TOOL activity duy nhất cho mỗi câu hỏi
- Nếu tool trả về total=0, DỪNG và trả lời "Không có hoạt động phù hợp"
- KHÔNG suy đoán hoặc tự tạo dữ liệu hoạt động
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
    """
    Process query - TỰ ĐỘNG INJECT TOKEN trước khi tools được gọi
    """
    try:
        # SET TOKEN GLOBAL NGAY TỪ ĐẦU
        if bearer_token:
            set_bearer_token(bearer_token)
            logger.info(f"[PROCESS] Token đã được set global: {bearer_token[:30]}...")
        else:
            logger.warning("[PROCESS] Không có bearer token - tools sẽ fail!")
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_role": user_role,
            "user_id": user_id,
            "bearer_token": bearer_token or ""  # Vẫn lưu để tracking
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
        
        # TRÍCH XUẤT TEXT PHẢN HỒI
        response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # TRÍCH XUẤT KẾT QUẢ TOOL
        activities_raw = []
        source = "general"
        total_activities = 0
        
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        logger.info(f"[DEBUG] Tìm thấy {len(tool_messages)} tool messages")
        
        if tool_messages:
            last_tool_msg = tool_messages[-1]
            logger.info(f"[DEBUG] Tool message cuối: {last_tool_msg.name if hasattr(last_tool_msg, 'name') else 'unknown'}")
            
            try:
                tool_result = json.loads(last_tool_msg.content) if isinstance(last_tool_msg.content, str) else last_tool_msg.content
                
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
        
        return json.dumps({
            "status": "success",
            "data": {
                "response": response_text,
                "user_role": user_role,
                "user_id": user_id,
                "source": source,
                "activities": activities_raw,
                "total_activities": total_activities
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