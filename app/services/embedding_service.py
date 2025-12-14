import os
import logging
import gc
from typing import List, Optional
from functools import lru_cache
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    USING_NEW_LANGCHAIN = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    USING_NEW_LANGCHAIN = False

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
import redis
import numpy as np
from app.config import Config
from app.services.document_loader import load_new_documents

logger = logging.getLogger(__name__)

# Global caches
_embedding_model_cache = None
_embedding_model_lock = None
_redis_client = None

try:
    import threading
    _embedding_model_lock = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _embedding_model_lock = DummyLock()

# UNIFIED INDEX (backward compatibility - giờ không dùng nữa)
UNIFIED_INDEX_NAME = "unified_documents_index"

def get_redis_client():
    """Get Redis client connection"""
    global _redis_client
    if _redis_client is None:
        try:
            host = os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', 6379))
            db = int(os.getenv('REDIS_DB', 0))
            password = os.getenv('REDIS_PASSWORD', None)
            
            logger.info(f"Connecting to Redis at {host}:{port}/{db}")
            
            _redis_client = redis.Redis(
                host=host, port=port, db=db, password=password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            _redis_client.ping()
            logger.info("Redis connection successful")
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    return _redis_client

def get_redis_url():
    """Get Redis connection URL"""
    host = os.getenv('REDIS_HOST', 'localhost')
    port = os.getenv('REDIS_PORT', '6379')
    db = os.getenv('REDIS_DB', '0')
    password = os.getenv('REDIS_PASSWORD', '')
    
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"

@lru_cache(maxsize=1)
def _create_embedding_model():
    """Create embedding model with caching"""
    logger.info("Creating new embedding model instance...")
    
    model = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={'device': 'cuda', 'trust_remote_code': True}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Embedding model created successfully")
    return model

def get_embedding_model():
    """Get embedding model singleton"""
    global _embedding_model_cache
    
    if _embedding_model_cache is None:
        with _embedding_model_lock:
            if _embedding_model_cache is None:
                _embedding_model_cache = _create_embedding_model()
    
    return _embedding_model_cache

def get_unit_redis_index(unit_name: str, embedding_dim: int = 768):
    """
    Tạo hoặc lấy Redis index riêng cho từng đơn vị
    """
    index_name = Config.get_unit_index_name(unit_name)
    
    schema = {
        "index": {
            "name": index_name,
            "prefix": f"doc:{index_name}",
            "storage_type": "hash"
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "doc_id", "type": "tag"},
            {"name": "filename", "type": "tag"},
            {"name": "uploaded_by", "type": "text"},
            {"name": "created_at", "type": "text"},
            {"name": "chunk_id", "type": "numeric"},
            {"name": "unit_name", "type": "tag"},  # Thêm unit_name field
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
        
        logger.info(f"Connecting to Redis for unit index: {index_name}")
        index.connect(redis_url)
        
        try:
            index.create(overwrite=False)
            logger.info(f"Unit Redis index '{index_name}' created successfully")
        except Exception as create_error:
            if "already exists" in str(create_error).lower():
                logger.info(f"Unit Redis index '{index_name}' already exists")
            else:
                raise
        
        return index
        
    except Exception as e:
        logger.error(f"Error creating unit Redis index: {e}")
        raise

def semantic_sliding_window_split(text: str, embedding_model, window_overlap: float = 0.2) -> List[str]:
    """Sliding window với semantic boundaries"""
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
    """Get text splitter"""
    try:
        if use_semantic:
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
            return RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    except Exception as e:
        logger.warning(f"Failed to create semantic splitter: {e}")
        return RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

def add_to_embedding(file_path: str, metadata, unit_name: str, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2):
    """
    Thêm documents vào unit-based Redis index
    """
    try:
        logger.info(f"Starting embedding for: {file_path} (unit: {unit_name})")
        
        redis_client = get_redis_client()
        documents = load_new_documents(file_path, metadata)
        
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return False

        embedding_model = get_embedding_model()
        text_splitter = get_text_splitter(
            use_semantic=use_semantic_chunking, 
            semantic_overlap=semantic_overlap,
            embedding_model=embedding_model
        )
        
        try:
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
        except Exception as e:
            if use_semantic_chunking:
                logger.warning(f"Semantic chunking failed: {e}. Fallback to traditional")
                text_splitter = get_text_splitter(use_semantic=False)
                chunks = text_splitter.split_documents(documents)
            else:
                raise
        
        if not chunks:
            return False
        
        # Lấy unit-based index
        embedding_dim = len(embedding_model.embed_query("test"))
        index = get_unit_redis_index(unit_name, embedding_dim)
        index_name = Config.get_unit_index_name(unit_name)
        
        metadata_dict = metadata.dict(by_alias=True)
        doc_id = metadata_dict.get('_id', metadata.id if hasattr(metadata, 'id') else '')
        
        # Thêm chunks vào unit index
        for i, chunk in enumerate(chunks):
            doc_key = f"doc:{index_name}:{doc_id}:{i}"
            embedding_vector = embedding_model.embed_query(chunk.page_content)
            
            redis_client.hset(doc_key, mapping={
                "content": chunk.page_content,
                "doc_id": doc_id,
                "filename": metadata.filename,
                "uploaded_by": metadata.uploaded_by,
                "created_at": metadata.createdAt,
                "chunk_id": i,
                "unit_name": unit_name,  # Lưu unit_name
                "embedding": np.array(embedding_vector, dtype=np.float32).tobytes()
            })
        
        logger.info(f"Added {len(chunks)} chunks to unit index '{index_name}' for doc: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error in add_to_embedding: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        gc.collect()

def delete_from_unit_index(doc_id: str, unit_name: str) -> bool:
    """Xóa document từ unit index"""
    try:
        redis_client = get_redis_client()
        index_name = Config.get_unit_index_name(unit_name)
        
        pattern = f"doc:{index_name}:{doc_id}:*"
        keys_to_delete = list(redis_client.scan_iter(match=pattern))
        
        if keys_to_delete:
            redis_client.delete(*keys_to_delete)
            logger.info(f"Deleted {len(keys_to_delete)} chunks from unit index '{index_name}'")
        else:
            logger.warning(f"No chunks found for doc_id: {doc_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting from unit index: {e}")
        return False

def update_metadata_only(doc_id: str, new_metadata, unit_name: str) -> bool:
    """Update only metadata trong unit index"""
    try:
        redis_client = get_redis_client()
        index_name = Config.get_unit_index_name(unit_name)
        
        pattern = f"doc:{index_name}:{doc_id}:*"
        updated_count = 0
        
        for key in redis_client.scan_iter(match=pattern):
            redis_client.hset(key, mapping={
                "filename": new_metadata.filename,
                "uploaded_by": new_metadata.uploaded_by,
            })
            updated_count += 1
        
        if updated_count > 0:
            logger.info(f"Updated metadata for {updated_count} chunks")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")
        return False

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, unit_name: str, force_re_embed: bool = False) -> bool:
    """Smart update với unit support"""
    try:
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if filename_changed or force_re_embed:
            delete_from_unit_index(doc_id, unit_name)
            file_path = new_metadata.url
            if os.path.exists(file_path):
                return add_to_embedding(file_path, new_metadata, unit_name)
            return False
        else:
            return update_metadata_only(doc_id, new_metadata, unit_name)
            
    except Exception as e:
        logger.error(f"Error in smart update: {e}")
        return False

# Backward compatibility functions
def delete_from_unified_index(doc_id: str) -> bool:
    """Deprecated - use delete_from_unit_index instead"""
    logger.warning("delete_from_unified_index is deprecated")
    return delete_from_unit_index(doc_id, "default_unit")