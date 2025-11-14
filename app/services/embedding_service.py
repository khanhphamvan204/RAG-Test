import os
import logging
import gc
from typing import List, Optional
from functools import lru_cache
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Text
from app.config import Config
from app.services.file_service import get_file_paths
from app.services.metadata_service import find_document_info
from app.services.document_loader import load_new_documents

# FIX 1: Import HuggingFaceEmbeddings từ package mới
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    USING_NEW_LANGCHAIN = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    USING_NEW_LANGCHAIN = False

import redis
import numpy as np
import json

logger = logging.getLogger(__name__)

# Global embedding model cache
_embedding_model_cache = None
_embedding_model_lock = None
_redis_client = None

# UNIFIED INDEX NAME
UNIFIED_INDEX_NAME = "unified_documents_index"

try:
    import threading
    _embedding_model_lock = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _embedding_model_lock = DummyLock()

def get_redis_client():
    """Get Redis client connection with proper error handling"""
    global _redis_client
    if _redis_client is None:
        try:
            # FIX 2: Lấy từ environment variables với fallback
            host = os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', 6379))
            db = int(os.getenv('REDIS_DB', 0))
            password = os.getenv('REDIS_PASSWORD', None)
            
            logger.info(f"Connecting to Redis at {host}:{port}/{db}")
            
            _redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                socket_connect_timeout=5,  # Timeout 5s
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            _redis_client.ping()
            logger.info("Redis connection successful")
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.error(f"Please check your Redis configuration:")
            logger.error(f"  - REDIS_HOST={os.getenv('REDIS_HOST', 'localhost')}")
            logger.error(f"  - REDIS_PORT={os.getenv('REDIS_PORT', '6379')}")
            logger.error(f"  - Is Redis server running?")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            raise
    
    return _redis_client

def get_redis_url():
    """Get Redis connection URL for RedisVL"""
    host = os.getenv('REDIS_HOST', 'localhost')
    port = os.getenv('REDIS_PORT', '6379')
    db = os.getenv('REDIS_DB', '0')
    password = os.getenv('REDIS_PASSWORD', '')
    
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"

@lru_cache(maxsize=1)
def _create_embedding_model():
    """Private function to create embedding model with caching"""
    logger.info("Creating new embedding model instance...")
    
    if USING_NEW_LANGCHAIN:
        logger.info("Using new langchain-huggingface package")
    else:
        logger.warning("Using deprecated langchain-community. Consider upgrading: pip install -U langchain-huggingface")
    
    model = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Embedding model created successfully")
    return model

def get_embedding_model():
    """Get consistent embedding model with singleton pattern"""
    global _embedding_model_cache
    
    if _embedding_model_cache is None:
        with _embedding_model_lock:
            if _embedding_model_cache is None:
                _embedding_model_cache = _create_embedding_model()
    
    return _embedding_model_cache

def clear_embedding_model_cache():
    """Clear embedding model cache - useful for memory management"""
    global _embedding_model_cache
    with _embedding_model_lock:
        if _embedding_model_cache is not None:
            logger.info("Clearing embedding model cache")
            _embedding_model_cache = None
            _create_embedding_model.cache_clear()
            gc.collect()

class EmbeddingModelManager:
    """Context manager for embedding model to ensure cleanup"""
    def __init__(self):
        self.model = None
    
    def __enter__(self):
        self.model = get_embedding_model()
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def semantic_sliding_window_split(text: str, embedding_model, window_overlap: float = 0.2) -> List[str]:
    """Sliding window với tỷ lệ overlap dựa trên semantic boundaries"""
    try:
        semantic_chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            sentence_split_regex=r'(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b\d\d\d\.)(?<!\b[A-Za-zÀ-ỹ]\.)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ]\.)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ][A-Za-zÀ-ỹ]\.)(?<=[.?!…])\s+(?=[A-ZÀ-Ỵ0-9])',
            buffer_size=5,
        )
        
        chunks = semantic_chunker.split_text(text)
        
        if len(chunks) <= 1:
            return chunks
        
        sliding_chunks = []
        
        for i in range(len(chunks)):
            if i == 0:
                sliding_chunks.append(chunks[i])
            else:
                prev_chunk_words = chunks[i-1].split()
                overlap_words_count = int(len(prev_chunk_words) * window_overlap)
                
                if overlap_words_count > 0:
                    overlap_text = ' '.join(prev_chunk_words[-overlap_words_count:])
                    new_chunk = overlap_text + " " + chunks[i]
                else:
                    new_chunk = chunks[i]
                
                sliding_chunks.append(new_chunk)
        
        return sliding_chunks
        
    except Exception as e:
        logger.warning(f"Semantic sliding window failed: {e}")
        return [text]

def get_text_splitter(use_semantic: bool = True, semantic_overlap: float = 0.2, embedding_model=None):
    """Get text splitter - semantic or traditional"""
    try:
        if use_semantic:
            logger.info("Using SemanticChunker with sliding window overlap for text splitting")
            
            if embedding_model is None:
                embedding_model = get_embedding_model()
            
            class SemanticSlidingWindowSplitter:
                def __init__(self, embedding_model, window_overlap=0.2):
                    self.embedding_model = embedding_model
                    self.window_overlap = window_overlap
                
                def split_text(self, text: str) -> List[str]:
                    return semantic_sliding_window_split(text, self.embedding_model, self.window_overlap)
                
                def split_documents(self, documents: List[Document]) -> List[Document]:
                    chunks = []
                    for doc in documents:
                        text_chunks = self.split_text(doc.page_content)
                        for chunk_text in text_chunks:
                            chunk_doc = Document(
                                page_content=chunk_text,
                                metadata=doc.metadata.copy()
                            )
                            chunks.append(chunk_doc)
                    return chunks
            
            return SemanticSlidingWindowSplitter(embedding_model, semantic_overlap)
        else:
            logger.info("Using RecursiveCharacterTextSplitter for text splitting")
            return RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=200
            )
    except Exception as e:
        logger.warning(f"Failed to create SemanticSlidingWindowSplitter: {e}")
        logger.info("Falling back to RecursiveCharacterTextSplitter")
        return RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )

def get_unified_redis_index(embedding_dim: int = 768):
    """
    Tạo hoặc lấy index Redis duy nhất cho toàn bộ hệ thống.
    With improved error handling.
    """
    schema = {
        "index": {
            "name": UNIFIED_INDEX_NAME,
            "prefix": f"doc:{UNIFIED_INDEX_NAME}",
            "storage_type": "hash"
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "doc_id", "type": "tag"},
            {"name": "filename", "type": "tag"},
            {"name": "uploaded_by", "type": "text"},
            {"name": "created_at", "type": "text"},
            {"name": "chunk_id", "type": "numeric"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": embedding_dim,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32"
                }
            }
        ]
    }
    
    try:
        index = SearchIndex.from_dict(schema)
        redis_url = get_redis_url()
        
        logger.info(f"Connecting to Redis for index at: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
        index.connect(redis_url)
        
        # Try to create index, nếu đã tồn tại thì skip
        try:
            index.create(overwrite=False)
            logger.info(f"Unified Redis index '{UNIFIED_INDEX_NAME}' created successfully")
        except Exception as create_error:
            if "already exists" in str(create_error).lower():
                logger.info(f"Unified Redis index '{UNIFIED_INDEX_NAME}' already exists")
            else:
                raise
        
        return index
        
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        logger.error("Please ensure Redis is running and accessible")
        raise
    except Exception as e:
        logger.error(f"Error creating unified Redis index: {e}")
        raise

def add_to_embedding(file_path: str, metadata, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2):
    """
    Thêm documents vào unified Redis index.
    With improved error handling.
    """
    try:
        logger.info(f"Starting embedding process for: {file_path}")
        
        # Kiểm tra Redis connection trước
        try:
            redis_client = get_redis_client()
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Redis: {e}")
            return False
        
        documents = load_new_documents(file_path, metadata)
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return False

        with EmbeddingModelManager() as embedding_model:
            text_splitter = get_text_splitter(
                use_semantic=use_semantic_chunking, 
                semantic_overlap=semantic_overlap,
                embedding_model=embedding_model
            )
            
            try:
                chunks = text_splitter.split_documents(documents)
                if use_semantic_chunking:
                    logger.info(f"Successfully created {len(chunks)} chunks using semantic chunking")
                else:
                    logger.info(f"Successfully created {len(chunks)} chunks using traditional chunking")
            except Exception as e:
                if use_semantic_chunking:
                    logger.warning(f"Semantic chunking failed: {e}. Falling back to traditional chunking")
                    text_splitter = get_text_splitter(use_semantic=False)
                    chunks = text_splitter.split_documents(documents)
                    logger.info(f"Created {len(chunks)} chunks using fallback traditional chunking")
                else:
                    raise e
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Sử dụng UNIFIED INDEX
            embedding_dim = len(embedding_model.embed_query("test"))
            index = get_unified_redis_index(embedding_dim)
            
            metadata_dict = metadata.dict(by_alias=True)
            doc_id = metadata_dict.get('_id', metadata.id if hasattr(metadata, 'id') else '')
            
            # Thêm chunks vào unified index
            for i, chunk in enumerate(chunks):
                doc_key = f"doc:{UNIFIED_INDEX_NAME}:{doc_id}:{i}"
                embedding_vector = embedding_model.embed_query(chunk.page_content)
                
                redis_client.hset(doc_key, mapping={
                    "content": chunk.page_content,
                    "doc_id": doc_id,
                    "filename": metadata.filename,
                    "uploaded_by": metadata.uploaded_by,
                    "created_at": metadata.createdAt,
                    "chunk_id": i,
                    "embedding": np.array(embedding_vector, dtype=np.float32).tobytes()
                })
            
            logger.info(f"Successfully added {len(chunks)} chunks to unified index for doc_id: {doc_id}")
            return True
            
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error in add_to_embedding: {e}")
        logger.error("Please check if Redis server is running")
        return False
    except Exception as e:
        logger.error(f"Error in add_to_embedding: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        gc.collect()

def delete_from_unified_index(doc_id: str) -> bool:
    """Xóa tất cả chunks của một document từ unified index"""
    try:
        redis_client = get_redis_client()
        
        pattern = f"doc:{UNIFIED_INDEX_NAME}:{doc_id}:*"
        keys_to_delete = []
        
        for key in redis_client.scan_iter(match=pattern):
            keys_to_delete.append(key)
        
        if keys_to_delete:
            redis_client.delete(*keys_to_delete)
            logger.info(f"Deleted {len(keys_to_delete)} chunks for doc_id: {doc_id}")
        else:
            logger.warning(f"No chunks found for doc_id: {doc_id}")
        
        return True
        
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error deleting from unified index: {str(e)}")
        return False

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2) -> bool:
    """Update document by re-embedding vào unified index"""
    try:
        success = delete_from_unified_index(doc_id)
        if not success:
            logger.warning(f"Failed to delete old chunks for doc_id: {doc_id}")
        
        file_path = new_metadata.url
        if os.path.exists(file_path):
            return add_to_embedding(file_path, new_metadata, use_semantic_chunking, semantic_overlap)
        else:
            logger.error(f"File not found for re-embedding: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata in vector store: {str(e)}")
        return False

def update_metadata_only(doc_id: str, new_metadata) -> bool:
    """Update only metadata trong unified index"""
    try:
        redis_client = get_redis_client()
        
        pattern = f"doc:{UNIFIED_INDEX_NAME}:{doc_id}:*"
        updated_count = 0
        
        for key in redis_client.scan_iter(match=pattern):
            redis_client.hset(key, mapping={
                "filename": new_metadata.filename,
                "uploaded_by": new_metadata.uploaded_by,
            })
            updated_count += 1
        
        if updated_count > 0:
            logger.info(f"Updated metadata for {updated_count} chunks of document: {doc_id}")
            return True
        else:
            logger.warning(f"No chunks found for document: {doc_id}")
            return False
            
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error updating metadata only: {str(e)}")
        return False

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, force_re_embed: bool = False, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2) -> bool:
    """Smart metadata update với unified index"""
    try:
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if filename_changed or force_re_embed:
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, semantic_overlap)
        else:
            success = update_metadata_only(doc_id, new_metadata)
            if not success:
                logger.info("Metadata-only update failed, attempting re-embedding")
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, semantic_overlap)
            return success
            
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False

def get_embedding_model_info():
    """Get information about current embedding model"""
    global _embedding_model_cache
    return {
        "is_cached": _embedding_model_cache is not None,
        "cache_info": _create_embedding_model.cache_info() if hasattr(_create_embedding_model, 'cache_info') else None,
        "unified_index_name": UNIFIED_INDEX_NAME,
        "using_new_langchain": USING_NEW_LANGCHAIN
    }

def cleanup_embedding_resources():
    """Cleanup embedding resources on shutdown"""
    logger.info("Cleaning up embedding resources...")
    clear_embedding_model_cache()

def test_redis_connection():
    """Test Redis connection - useful for debugging"""
    try:
        redis_client = get_redis_client()
        redis_client.ping()
        logger.info("Redis connection test successful")
        
        # Test index existence
        pattern = f"doc:{UNIFIED_INDEX_NAME}:*"
        sample_keys = list(redis_client.scan_iter(match=pattern, count=1))
        logger.info(f"Unified index documents found: {len(sample_keys) > 0}")
        
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False