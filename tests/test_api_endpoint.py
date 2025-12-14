"""
Test RAG system qua API endpoint /chat/process-query
"""
import pytest
import requests
import json
from typing import Dict


# Config
BASE_URL = "http://localhost:8000"
PROCESS_QUERY_ENDPOINT = f"{BASE_URL}/chat/process-query"


@pytest.fixture
def test_bearer_token():
    """Mock bearer token cho testing"""
    return "test_token_12345"


@pytest.fixture
def test_unit_name():
    """Test unit name"""
    return "khoa_cong_nghe_thong_tin"


def make_chat_request(query: str, bearer_token: str, unit_name: str, user_id: int = 1) -> Dict:
    """Helper function để gọi API endpoint"""
    payload = {
        "query": query,
        "user_role": "student",
        "user_id": user_id,
        "bearer_token": bearer_token,
        "unit_name": unit_name
    }
    
    response = requests.post(PROCESS_QUERY_ENDPOINT, json=payload, timeout=30)
    return response


def test_api_endpoint_available():
    """Test API endpoint có hoạt động không"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.skip("API server is not running. Start with: uvicorn main:app --reload")


def test_rag_query_via_api(test_bearer_token, test_unit_name):
    """Test RAG query qua API endpoint"""
    query = "Quy định về điểm rèn luyện là gì?"
    
    response = make_chat_request(query, test_bearer_token, test_unit_name)
    
    # Check response status
    assert response.status_code == 200
    
    # Parse response
    data = response.json()
    
    # Validate response structure
    assert "status" in data
    assert "data" in data
    
    if data["status"] == "success":
        response_data = data["data"]
        assert "response" in response_data
        assert "source" in response_data
        assert "unit_name" in response_data
        
        # Response should not be empty
        assert len(response_data["response"]) > 0
        
        # Unit name should match
        assert response_data["unit_name"] == test_unit_name


def test_multiple_queries_via_api(test_bearer_token, test_unit_name):
    """Test nhiều queries liên tiếp"""
    queries = [
        "Điều kiện tốt nghiệp là gì?",
        "Làm thế nào để đăng ký học phần?",
        "Quy định về học lại môn học?"
    ]
    
    for query in queries:
        response = make_chat_request(query, test_bearer_token, test_unit_name)
        assert response.status_code == 200
        
        data = response.json()
        if data["status"] == "success":
            assert len(data["data"]["response"]) > 10  # Meaningful response


def test_conversation_thread_via_api(test_bearer_token, test_unit_name):
    """Test conversation với thread_id để maintain context"""
    thread_id = "test_thread_001"
    
    # First query
    payload1 = {
        "query": "Điểm rèn luyện là gì?",
        "user_role": "student",
        "user_id": 1,
        "bearer_token": test_bearer_token,
        "unit_name": test_unit_name,
        "thread_id": thread_id
    }
    
    response1 = requests.post(PROCESS_QUERY_ENDPOINT, json=payload1, timeout=30)
    assert response1.status_code == 200
    
    # Follow-up query
    payload2 = {
        "query": "Và cách tính điểm đó như thế nào?",
        "user_role": "student", 
        "user_id": 1,
        "bearer_token": test_bearer_token,
        "unit_name": test_unit_name,
        "thread_id": thread_id
    }
    
    response2 = requests.post(PROCESS_QUERY_ENDPOINT, json=payload2, timeout=30)
    assert response2.status_code == 200
    
    data2 = response2.json()
    if data2["status"] == "success":
        # Should have context from previous query
        assert "context_info" in data2["data"]
        assert data2["data"]["context_info"]["total_messages"] > 2


@pytest.mark.parametrize("query_type,query", [
    ("rag", "Quy định học vụ là gì?"),
    ("rag", "Điều kiện tốt nghiệp?"),
    ("general", "Xin chào"),
])
def test_different_query_types(query_type, query, test_bearer_token, test_unit_name):
    """Test các loại queries khác nhau"""
    response = make_chat_request(query, test_bearer_token, test_unit_name)
    assert response.status_code == 200
    
    data = response.json()
    if data["status"] == "success":
        assert "source" in data["data"]
        # Source có thể là 'rag', 'activity', hoặc 'general'


def test_api_response_time(test_bearer_token, test_unit_name):
    """Test response time của API"""
    import time
    
    query = "Test query"
    start_time = time.time()
    
    response = make_chat_request(query, test_bearer_token, test_unit_name)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 10  # Should respond within 10 seconds


def test_api_error_handling(test_unit_name):
    """Test API error handling với invalid inputs"""
    # Missing required fields
    payload = {
        "query": "Test"
        # Missing other required fields
    }
    
    response = requests.post(PROCESS_QUERY_ENDPOINT, json=payload, timeout=10)
    # Should handle gracefully
    assert response.status_code in [200, 400, 422]
