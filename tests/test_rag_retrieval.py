"""
Test RAG retrieval quality
"""
import pytest
from app.services.rag_service import rag_service
from app.models.vector_models import VectorSearchRequest


def test_rag_search_returns_results(sample_query):
    """Test that RAG search returns results"""
    result = rag_service.search_with_llm(
        VectorSearchRequest(
            query=sample_query,
            k=5,
            similarity_threshold=0.3
        ),
        unit_name="default_unit"
    )
    
    assert result is not None
    assert result.llm_response is not None
    assert len(result.llm_response) > 0


def test_rag_search_unit_specific():
    """Test unit-specific search"""
    result = rag_service.search_with_llm(
        VectorSearchRequest(
            query="quy định học vụ",
            k=3,
            similarity_threshold=0.3
        ),
        unit_name="default_unit"
    )
    
    assert result.unit_name == "default_unit"


def test_similarity_threshold_filtering():
    """Test that similarity threshold works"""
    # High threshold should return fewer results
    result_high = rag_service.search_with_llm(
        VectorSearchRequest(
            query="test query",
            k=10,
            similarity_threshold=0.9
        ),
        unit_name="default_unit"
    )
    
    # Low threshold should be more permissive
    result_low = rag_service.search_with_llm(
        VectorSearchRequest(
            query="test query",
            k=10,
            similarity_threshold=0.1
        ),
        unit_name="default_unit"
    )
    
    # Both should return valid responses
    assert result_high is not None
    assert result_low is not None


@pytest.mark.parametrize("k_value", [1, 3, 5, 10])
def test_different_k_values(k_value, sample_query):
    """Test different k values for retrieval"""
    result = rag_service.search_with_llm(
        VectorSearchRequest(
            query=sample_query,
            k=k_value,
            similarity_threshold=0.3
        ),
        unit_name="default_unit"
    )
    
    assert result is not None
    assert result.llm_response is not None
