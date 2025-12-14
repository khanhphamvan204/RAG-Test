"""
Test answer generation quality
"""
import pytest
import json
from pathlib import Path


@pytest.fixture
def sample_questions(sample_questions_file):
    """Load sample questions for testing"""
    if not sample_questions_file.exists():
        pytest.skip("Sample questions file not found. Run generate_test_data.py first.")
    
    with open(sample_questions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_answer_not_empty(sample_questions):
    """Test that generated answers are not empty"""
    from app.services.rag_service import rag_service
    from app.models.vector_models import VectorSearchRequest
    
    # Test with first question
    if sample_questions:
        question = sample_questions[0]['question']
        unit_name = sample_questions[0].get('unit_name', 'default_unit')
        
        result = rag_service.search_with_llm(
            VectorSearchRequest(
                query=question,
                k=5,
                similarity_threshold=0.3
            ),
            unit_name=unit_name
        )
        
        assert result.llm_response
        assert len(result.llm_response) > 10  # At least some meaningful content


def test_answer_format(sample_questions):
    """Test that answer follows expected format"""
    from app.services.rag_service import rag_service
    from app.models.vector_models import VectorSearchRequest
    
    if sample_questions:
        question = sample_questions[0]['question']
        unit_name = sample_questions[0].get('unit_name', 'default_unit')
        
        result = rag_service.search_with_llm(
            VectorSearchRequest(
                query=question,
                k=5,
                similarity_threshold=0.3
            ),
            unit_name=unit_name
        )
        
        # Answer should not contain error messages
        assert "Lỗi hệ thống" not in result.llm_response
        assert "Error" not in result.llm_response


def test_multiple_questions(sample_questions):
    """Test RAG with multiple questions"""
    from app.services.rag_service import rag_service
    from app.models.vector_models import VectorSearchRequest
    
    # Test first 3 questions
    test_subset = sample_questions[:3] if len(sample_questions) >= 3 else sample_questions
    
    for item in test_subset:
        question = item['question']
        unit_name = item.get('unit_name', 'default_unit')
        
        result = rag_service.search_with_llm(
            VectorSearchRequest(
                query=question,
                k=5,
                similarity_threshold=0.3
            ),
            unit_name=unit_name
        )
        
        # Each question should get a response
        assert result is not None
        assert result.llm_response
        assert len(result.llm_response) > 0


def test_answer_consistency():
    """Test that same question returns consistent answers"""
    from app.services.rag_service import rag_service
    from app.models.vector_models import VectorSearchRequest
    
    query = "Quy định về điểm rèn luyện"
    
    # Run twice
    result1 = rag_service.search_with_llm(
        VectorSearchRequest(query=query, k=5, similarity_threshold=0.3),
        unit_name="default_unit"
    )
    
    result2 = rag_service.search_with_llm(
        VectorSearchRequest(query=query, k=5, similarity_threshold=0.3),
        unit_name="default_unit"
    )
    
    # Both should return valid responses
    assert result1.llm_response
    assert result2.llm_response
    
    # Responses should have similar length (allowing some variation due to LLM randomness)
    len_ratio = len(result1.llm_response) / len(result2.llm_response)
    assert 0.5 < len_ratio < 2.0  # Within 2x difference
