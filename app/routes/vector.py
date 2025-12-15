import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.services.embedding_service import (
    add_to_embedding, delete_from_unit_index, smart_metadata_update,
    get_embedding_model, get_redis_client
)
from app.services.langgraph_service import process_query
from app.services.metadata_service import save_metadata, delete_metadata, find_document_info
from app.services.file_service import get_file_paths
from app.services.auth_service import verify_token_v2, verify_admin_or_advisor_role
from app.config import Config
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import os
import json
import uuid
from datetime import datetime, timezone, timedelta
import shutil
import logging
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
import time
import numpy as np
from dotenv import load_dotenv
from app.models.vector_models import (
    AddVectorRequest,
    SearchResult,
    VectorSearchRequest,
    VectorSearchResponse
)

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

router = APIRouter()
logger = logging.getLogger(__name__)

def standardization(distance: float) -> float:
    """Chuyển đổi cosine distance thành similarity score"""
    return 1 - distance

@router.post("/add", response_model=dict)
async def add_vector_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form(...),
    current_user: dict = Depends(verify_admin_or_advisor_role)  # Chỉ admin/advisor mới thêm được
):
    """
    Thêm tài liệu vào thư viện của đơn vị
    """
    try:
        # Lấy unit_name từ token
        unit_name = current_user.get('unit_name', 'default_unit')
        user_id = current_user.get('id')
        user_role = current_user.get('role')
        
        logger.info(f"[ADD] User {user_id} ({user_role}) from unit '{unit_name}' adding file")
        
        file_name = file.filename
        
        # Lấy file paths theo unit
        file_path, vector_db_path = get_file_paths(file_name, unit_name)
        
        if os.path.exists(file_path):
            raise HTTPException(
                status_code=409, 
                detail=f"File already exists in unit '{unit_name}': {file_path}"
            )
        
        supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file_name.lower())[1]
        if file_extension not in supported_extensions:
            raise HTTPException(status_code=400, detail=f"File format {file_extension} not supported")
        
        generated_id = str(uuid.uuid4())
        vietnam_tz = timezone(timedelta(hours=7))
        created_at = datetime.now(vietnam_tz).isoformat()
        
        file_url = file_path
        
        metadata = AddVectorRequest(
            _id=generated_id,
            filename=file_name,
            url=file_url,
            uploaded_by=uploaded_by,
            createdAt=created_at,
            unit_name=unit_name  # Thêm unit_name vào metadata
        )
        
        # Tạo thư mục và lưu file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            save_metadata(metadata)
            add_to_embedding(file_path, metadata, unit_name)
        except Exception as embed_error:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process embeddings: {str(embed_error)}"
            )
        
        index_name = Config.get_unit_index_name(unit_name)
        
        return {
            "message": "Vector added successfully",
            "_id": generated_id,
            "filename": file_name,
            "file_path": file_path,
            "vector_index": index_name,
            "unit_name": unit_name,
            "status": "created"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.delete("/{doc_id}", response_model=dict)
async def delete_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_admin_or_advisor_role)
):
    """
    Xóa tài liệu (chỉ trong unit của user)
    """
    try:
        unit_name = current_user.get('unit_name', 'default_unit')
        user_role = current_user.get('role')
        
        doc_info = find_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document not found")
        
        # Kiểm tra quyền: chỉ xóa được tài liệu trong unit của mình
        doc_unit = doc_info.get('unit_name', 'default_unit')
        if doc_unit != unit_name and user_role != 'admin':  # Admin có thể xóa mọi unit
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: Document belongs to different unit"
            )
        
        filename = doc_info.get('filename')
        file_path = doc_info.get('url')
        
        deletion_results = {
            "file_deleted": False,
            "metadata_deleted": False,
            "vector_deleted": False
        }
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            deletion_results["file_deleted"] = True
        
        deletion_results["vector_deleted"] = delete_from_unit_index(doc_id, doc_unit)
        deletion_results["metadata_deleted"] = delete_metadata(doc_id)
        
        message = "Document deleted successfully" if all(deletion_results.values()) else "Document partially deleted"
        
        return {
            "message": message,
            "_id": doc_id,
            "filename": filename,
            "deletion_results": deletion_results,
            "unit_name": doc_unit,
            "vector_index": Config.get_unit_index_name(doc_unit)
        }
        
    except Exception as e:
        logger.error(f"Error deleting document: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/{doc_id}", response_model=dict)
async def get_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2)
):
    """
    Lấy thông tin document
    """
    try:
        unit_name = current_user.get('unit_name', 'default_unit')
        user_role = current_user.get('role')
        
        doc_info = find_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document not found")
        
        # Kiểm tra quyền truy cập
        doc_unit = doc_info.get('unit_name', 'default_unit')
        if doc_unit != unit_name and user_role not in ['admin', 'advisor']:
            raise HTTPException(status_code=403, detail="Access denied")
        
        file_path = doc_info.get('url')
        file_exists = os.path.exists(file_path) if file_path else False
        
        # Kiểm tra chunks trong unit index
        redis_client = get_redis_client()
        index_name = Config.get_unit_index_name(doc_unit)
        pattern = f"doc:{index_name}:{doc_id}:*"
        chunk_count = len(list(redis_client.scan_iter(match=pattern, count=1000)))
        
        file_size = os.path.getsize(file_path) if file_exists else None
        
        return {
            **doc_info,
            "file_exists": file_exists,
            "vector_exists": chunk_count > 0,
            "chunk_count": chunk_count,
            "vector_index": index_name,
            "file_size": file_size,
            "unit_name": doc_unit
        }
        
    except Exception as e:
        logger.error(f"Error getting document: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/search", response_model=VectorSearchResponse)
async def search_vector_documents(
    request: VectorSearchRequest,
    current_user: dict = Depends(verify_token_v2) 
):
    """
    Search CHUNG trong unit index của user
    """
    start_time = time.time()
    
    try:
        # Lấy unit_name từ token
        unit_name = current_user.get('unit_name', 'default_unit')
        user_role = current_user.get('role')
        
        logger.info(f"[SEARCH] User from unit '{unit_name}' searching: {request.query}")
        
        embedding_model = get_embedding_model()
        redis_client = get_redis_client()
        
        # Lấy index name của unit
        index_name = Config.get_unit_index_name(unit_name)
        
        # Kiểm tra documents trong unit index
        pattern = f"doc:{index_name}:*"
        sample_keys = list(redis_client.scan_iter(match=pattern, count=1))
        
        if not sample_keys:
            return VectorSearchResponse(
                query=request.query,
                results=[],
                total_found=0,
                k_requested=request.k,
                similarity_threshold=request.similarity_threshold,
                search_time_ms=round((time.time() - start_time) * 1000, 2),
                unit_name=unit_name
            )
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(request.query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        # Vector query
        v = VectorQuery(
            vector=query_vector.tolist(),
            vector_field_name="embedding",
            return_fields=["content", "doc_id", "filename", "uploaded_by", "created_at", "chunk_id", "unit_name"],
            num_results=request.k * 2
        )
        
        # Schema cho unit index
        from app.services.embedding_service import get_redis_url
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
                {"name": "unit_name", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": len(query_embedding),
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32"
                    }
                }
            ]
        }
        
        index = SearchIndex.from_dict(schema)
        index.connect(get_redis_url())
        
        # Execute search
        results = index.query(v)
        
        # Process results
        all_results = []
        for result in results:
            similarity = standardization(float(result.get('vector_distance', 1.0)))
            if similarity >= request.similarity_threshold:
                all_results.append({
                    "content": result.get('content', ''),
                    "metadata": {
                        "doc_id": result.get('doc_id', ''),
                        "filename": result.get('filename', ''),
                        "uploaded_by": result.get('uploaded_by', ''),
                        "created_at": result.get('created_at', ''),
                        "chunk_id": result.get('chunk_id', 0),
                        "similarity_score": similarity,
                        "unit_name": result.get('unit_name', unit_name)
                    }
                })
        
        all_results.sort(key=lambda x: x['metadata']['similarity_score'], reverse=True)
        top_results = all_results[:request.k]
        
        results_list = [
            SearchResult(content=result["content"], metadata=result["metadata"])
            for result in top_results
        ]
        
        search_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return VectorSearchResponse(
            query=request.query,
            results=results_list,
            total_found=len(results_list),
            k_requested=request.k,
            similarity_threshold=request.similarity_threshold,
            search_time_ms=search_time_ms,
            index_name=index_name,
            unit_name=unit_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

class ProcessQueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
    unit_name: str | None = "default_unit"

@router.post("/chat/process-query")
async def process_query_endpoint(
    request: ProcessQueryRequest,
    current_user: dict = Depends(verify_token_v2)
):
    """
    Process query với LangGraph - search trong unit của user
    """
    start_time = time.time()
    
    try:
        bearer_token = current_user.get('token')
        user_role = current_user.get('role', 'student')
        user_id = current_user.get('id', 0)
        unit_name = current_user.get('unit_name', 'default_unit')
        
        logger.info(f"[CHAT] User {user_id} from unit '{unit_name}' querying: {request.query}")
        
        # Truyền unit_name xuống process_query
        result_json = process_query(
            query=request.query,
            user_role=user_role,
            user_id=user_id,
            bearer_token=bearer_token,
            unit_name=unit_name,  # THÊM UNIT_NAME
            thread_id=request.thread_id
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[CHAT] Query processed in {elapsed:.2f}s")
        
        result_dict = json.loads(result_json)
        return result_dict
        
    except Exception as e:
        logger.error(f"[CHAT] Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


