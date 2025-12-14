"""
Test LangGraph integration and end-to-end flow
"""
import pytest
import json
from app.services.langgraph_service import process_query


def test_langgraph_rag_tool_selection():
    """Test that agent selects RAG tool for document questions"""
    response = process_query(
        query="Quy định về học vụ là gì?",
        user_role="student",
        user_id=1,
        unit_name="default_unit"
    )
    
    result = json.loads(response)
    
    assert result['status'] == 'success'
    assert result['data']['source'] in ['rag', 'general']
    assert result['data']['response']


def test_langgraph_unit_context():
    """Test that unit context is properly passed"""
    response = process_query(
        query="Có tài liệu nào không?",
        user_role="student",
        user_id=1,
        unit_name="test_unit"
    )
    
    result = json.loads(response)
    
    assert result['status'] == 'success'
    assert result['data']['unit_name'] == 'test_unit'


def test_langgraph_conversation_history():
    """Test multi-turn conversation"""
    thread_id = "test_thread_001"
    
    # First query
    response1 = process_query(
        query="Điểm rèn luyện là gì?",
        user_role="student",
        user_id=1,
        unit_name="default_unit",
        thread_id=thread_id
    )
    
    result1 = json.loads(response1)
    assert result1['status'] == 'success'
    assert result1['thread_id'] == thread_id
    
    # Follow-up query
    response2 = process_query(
        query="Làm sao để tính điểm đó?",
        user_role="student",
        user_id=1,
        unit_name="default_unit",
        thread_id=thread_id
    )
    
    result2 = json.loads(response2)
    assert result2['status'] == 'success'
    assert result2['thread_id'] == thread_id
    
    # Context should be maintained
    total_messages = result2['data']['context_info']['total_messages']
    assert total_messages > 2  # At least 2 exchanges


def test_langgraph_error_handling():
    """Test error handling in LangGraph"""
    # Query with invalid parameters should still return proper error
    response = process_query(
        query="",
        user_role="student",
        user_id=1,
        unit_name="default_unit"
    )
    
    result = json.loads(response)
    # Should handle gracefully
    assert 'status' in result


def test_context_window_management():
    """Test that context window is properly managed"""
    from app.services.langgraph_service import get_context_stats
    
    thread_id = "test_thread_context"
    
    # Make several queries
    for i in range(5):
        process_query(
            query=f"Câu hỏi số {i+1}",
            user_role="student",
            user_id=1,
            unit_name="default_unit",
            thread_id=thread_id
        )
    
    # Check context stats
    stats = get_context_stats(thread_id)
    
    assert 'total_messages' in stats
    assert 'human_messages' in stats
    assert 'ai_messages' in stats
