import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.services.embedding_service import (
    add_to_embedding, delete_from_unified_index, smart_metadata_update,
    get_embedding_model, get_redis_client, UNIFIED_INDEX_NAME
)
from app.services.langgraph_service import process_query
from app.services.metadata_service import save_metadata, delete_metadata, find_document_info
from app.services.file_service import get_file_paths
from app.services.auth_service import verify_token_v2
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import os
import json
import uuid
from datetime import datetime, timezone, timedelta
import shutil
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
import time
import numpy as np
from dotenv import load_dotenv
from app.models.vector_models import (
    AddVectorRequest,
    SearchResult,
    VectorSearchRequest,
    VectorSearchResponse
)
from fastapi.responses import JSONResponse
from typing import Literal, Optional
from app.services.activity_search_service import (
    ActivitySearchService,
    ActivitySearchRequest,
    ActivitySearchResponse,
    ActivitySearchWithLLMResponse
)
from fastapi import APIRouter, HTTPException, Depends, Header
activity_service = ActivitySearchService()

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
    current_user: dict = Depends(verify_token_v2)
):
    try:
        file_name = file.filename
        
        file_path, vector_db_path = get_file_paths(file_name)
        if os.path.exists(file_path):
            raise HTTPException(
                status_code=409, 
                detail=f"File already exists at path: {file_path}"
            )
        
        supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file_name.lower())[1]
        if file_extension not in supported_extensions:
            raise HTTPException(status_code=400, detail=f"File format {file_extension} not supported")
        
        generated_id = str(uuid.uuid4())
        vietnam_tz = timezone(timedelta(hours=7))
        created_at = datetime.now(vietnam_tz).isoformat()
        
        file_path, vector_db_path = get_file_paths(file_name)
        file_url = file_path
        
        metadata = AddVectorRequest(
            _id=generated_id,
            filename=file_name,
            url=file_url,
            uploaded_by=uploaded_by,
            createdAt=created_at
        )
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            save_metadata(metadata)
            add_to_embedding(file_path, metadata)
        except Exception as embed_error:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process embeddings: {str(embed_error)}"
            )
        
        return {
            "message": "Vector added successfully",
            "_id": generated_id,
            "filename": file_name,
            "file_path": file_path,
            "vector_index": UNIFIED_INDEX_NAME,  # Giờ trả về unified index name
            "status": "created"
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError as json_error:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in role fields: {str(json_error)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.delete("/{doc_id}", response_model=dict)
async def delete_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2)
):
    try:
        doc_info = find_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
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
        
        # Xóa từ unified index thay vì index riêng
        deletion_results["vector_deleted"] = delete_from_unified_index(doc_id)
        deletion_results["metadata_deleted"] = delete_metadata(doc_id)
        
        message = "Document deleted successfully" if all(deletion_results.values()) else "Document partially deleted"
        response = {
            "message": message,
            "_id": doc_id,
            "filename": filename,
            "deletion_results": deletion_results,
            "vector_index": UNIFIED_INDEX_NAME
        }
        
        if not all(deletion_results.values()):
            response["warning"] = "Some components could not be deleted"
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/{doc_id}", response_model=dict)
async def get_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2)
):
    try:
        doc_info = find_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        file_path = doc_info.get('url')
        file_exists = os.path.exists(file_path) if file_path else False
        
        # Kiểm tra chunks trong unified index
        redis_client = get_redis_client()
        pattern = f"doc:{UNIFIED_INDEX_NAME}:{doc_id}:*"
        chunk_count = len(list(redis_client.scan_iter(match=pattern, count=1000)))
        
        file_size = os.path.getsize(file_path) if file_exists else None
        
        return {
            **doc_info,
            "file_exists": file_exists,
            "vector_exists": chunk_count > 0,
            "chunk_count": chunk_count,
            "vector_index": UNIFIED_INDEX_NAME,
            "file_size": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")
    
@router.put("/{doc_id}", response_model=dict)
async def update_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2),
    filename: str = Form(None),
    uploaded_by: str = Form(None),
    force_re_embed: bool = Form(False)
):
    try:
        current_doc = find_document_info(doc_id)
        if not current_doc:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        old_metadata = current_doc.copy()
        current_filename = current_doc.get('filename')
        current_file_path = current_doc.get('url')
        
        final_filename = current_filename
        if filename:
            current_name, current_extension = os.path.splitext(current_filename)
            input_name, input_extension = os.path.splitext(filename)
            
            if input_extension:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Please provide filename without extension. Current file extension '{current_extension}' will be preserved automatically."
                )
            
            final_filename = filename + current_extension
            
            supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
            if current_extension.lower() not in supported_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Current file extension '{current_extension}' is not supported"
                )
        
        if filename and final_filename != current_filename:
            target_file_path, _ = get_file_paths(final_filename)
            if os.path.exists(target_file_path):
                raise HTTPException(
                    status_code=409,
                    detail=f"File '{final_filename}' already exists at path: {target_file_path}"
                )
        
        new_filename = final_filename
        new_uploaded_by = uploaded_by or current_doc.get('uploaded_by')
        filename_changed = filename and new_filename != current_filename
        
        operations = {
            "file_renamed": False,
            "vector_updated": False,
            "metadata_updated": False,
            "update_method": "none"
        }
        
        final_file_path = current_file_path
        if filename_changed:
            new_file_path, _ = get_file_paths(new_filename)
            if os.path.exists(current_file_path):
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                shutil.move(current_file_path, new_file_path)
                operations["file_renamed"] = True
                final_file_path = new_file_path
        
        new_metadata = AddVectorRequest(
            _id=doc_id,
            filename=new_filename,
            url=final_file_path,
            uploaded_by=new_uploaded_by,
            createdAt=current_doc.get('createdAt')
        )
        
        operations["vector_updated"] = smart_metadata_update(doc_id, old_metadata, new_metadata, force_re_embed)
        operations["update_method"] = "full_re_embed" if (filename_changed or force_re_embed) else "metadata_only"
        
        delete_metadata(doc_id)
        save_metadata(new_metadata)
        operations["metadata_updated"] = True
        
        response = {
            "message": "Document updated successfully" if operations["vector_updated"] and operations["metadata_updated"] else "Document partially updated",
            "_id": doc_id,
            "success": operations["vector_updated"] and operations["metadata_updated"],
            "updated_fields": {
                "filename": {"old": current_filename, "new": new_filename, "changed": filename_changed},
                "uploaded_by": {"old": current_doc.get('uploaded_by'), "new": new_uploaded_by, "changed": new_uploaded_by != current_doc.get('uploaded_by')},
            },
            "operations": operations,
            "paths": {
                "old_file_path": current_file_path,
                "new_file_path": final_file_path
            },
            "vector_index": UNIFIED_INDEX_NAME,
            "updatedAt": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "force_re_embed": force_re_embed
        }
        
        if not operations["vector_updated"] or not operations["metadata_updated"]:
            response["warnings"] = []
            if not operations["vector_updated"]:
                response["warnings"].append("Vector embeddings update failed")
            if not operations["metadata_updated"]:
                response["warnings"].append("Metadata database update failed")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")

@router.post("/search", response_model=VectorSearchResponse)
async def search_vector_documents(
    request: VectorSearchRequest,
    current_user: dict = Depends(verify_token_v2) 
):
    """
    Search trong unified index - đơn giản hơn nhiều!
    Không cần loop qua nhiều indexes nữa.
    """
    start_time = time.time()
    
    try:
        embedding_model = get_embedding_model()
        redis_client = get_redis_client()
        
        # Kiểm tra xem có documents nào không
        pattern = f"doc:{UNIFIED_INDEX_NAME}:*"
        sample_keys = list(redis_client.scan_iter(match=pattern, count=1))
        if not sample_keys:
            return VectorSearchResponse(
                query=request.query,
                results=[],
                total_found=0,
                k_requested=request.k,
                similarity_threshold=request.similarity_threshold,
                search_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(request.query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        # Tạo VectorQuery cho unified index
        v = VectorQuery(
            vector=query_vector.tolist(),
            vector_field_name="embedding",
            return_fields=["content", "doc_id", "filename", "uploaded_by", "created_at", "chunk_id"],
            num_results=request.k * 2  # Lấy nhiều hơn để filter sau
        )
        
        # Lấy schema cho unified index
        from app.services.embedding_service import get_redis_url
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
        
        # Execute search trên unified index
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
                        "similarity_score": similarity
                    }
                })
        
        # Sort và limit results
        all_results.sort(key=lambda x: x['metadata']['similarity_score'], reverse=True)
        top_results = all_results[:request.k]
        
        results = [
            SearchResult(
                content=result["content"], 
                metadata=result["metadata"]
            )
            for result in top_results
        ]
        
        search_time_ms = round((time.time() - start_time) * 1000, 2)
        return VectorSearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            k_requested=request.k,
            similarity_threshold=request.similarity_threshold,
            search_time_ms=search_time_ms,
            index_name=UNIFIED_INDEX_NAME
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")


class ProcessQueryRequest(BaseModel):
    query: str
    thread_id: str | None = None

class ProcessQueryResponse(BaseModel):
    status: str
    data: dict | None
    error: str | None
    thread_id: str | None


# ==================== HELPER FUNCTIONS ====================

def extract_bearer_token(authorization: str = Header(None)) -> str:
    """
    Extract Bearer token từ Authorization header
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )
    
    if not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Expected: Bearer <token>"
        )
    
    token = authorization.replace('Bearer ', '')
    return token



# app/routes/vector.py (trong file của bạn)

@router.post("/chat/process-query")
async def process_query_endpoint(
    request: ProcessQueryRequest,
    current_user: dict = Depends(verify_token_v2)
):
    """
    Process query voi LangGraph - ho tro activity search
    """
    start_time = time.time()
    
    try:
        # LAY TOKEN TU current_user
        logger.info(f"[ENDPOINT] current_user keys: {list(current_user.keys())}")
        logger.info(f"[ENDPOINT] current_user content: {current_user}")
        
        bearer_token = current_user.get('token')
        
        if not bearer_token:
            logger.error("[ENDPOINT] CRITICAL: No bearer token found in current_user!")
            logger.error(f"[ENDPOINT] Available keys: {list(current_user.keys())}")
            logger.error(f"[ENDPOINT] Full current_user: {current_user}")
        else:
            logger.info(f"[ENDPOINT] Bearer token found: {bearer_token[:30]}...")
        
        # Override user info tu token
        user_role = current_user.get('role', 'student')
        user_id = current_user.get('id', 0)
        
        logger.info(f"[ENDPOINT] Processing query from user_id={user_id}, role={user_role}")
        
        # TRUYEN TOKEN XUONG PROCESS_QUERY
        result_json = process_query(
            query=request.query,
            user_role=user_role,
            user_id=user_id,
            bearer_token=bearer_token,
            thread_id=request.thread_id
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[ENDPOINT] Query processed in {elapsed:.2f}s")
        
        result_dict = json.loads(result_json)
        return result_dict
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
