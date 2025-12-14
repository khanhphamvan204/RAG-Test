"""
Pytest configuration and fixtures for RAG testing
"""
import pytest
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


@pytest.fixture(scope="session")
def test_env():
    """Setup test environment variables"""
    return {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
        "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
    }


@pytest.fixture(scope="session")
def embedding_model():
    """Get embedding model instance"""
    from app.services.embedding_service import get_embedding_model
    return get_embedding_model()


@pytest.fixture(scope="session")
def redis_client():
    """Get Redis client instance"""
    from app.services.embedding_service import get_redis_client
    return get_redis_client()


@pytest.fixture(scope="function")
def sample_query():
    """Sample test query"""
    return "Quy định về điểm rèn luyện là gì?"


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_questions_file(test_data_dir):
    """Path to sample questions JSON file"""
    return test_data_dir / "sample_questions.json"
