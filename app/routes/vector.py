import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.services.embedding_service import (
    add_to_embedding, delete_from_redis_index, smart_metadata_update,
    get_embedding_model, get_redis_client
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
            "vector_db_path": vector_db_path,
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
        
        index_name = f"docs_{filename.replace('.', '_').replace(' ', '_')}"
        
        deletion_results = {
            "file_deleted": False,
            "metadata_deleted": False,
            "vector_deleted": False
        }
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            deletion_results["file_deleted"] = True
        
        deletion_results["vector_deleted"] = delete_from_redis_index(index_name, doc_id)
        deletion_results["metadata_deleted"] = delete_metadata(doc_id)
        
        message = "Document deleted successfully" if all(deletion_results.values()) else "Document partially deleted"
        response = {
            "message": message,
            "_id": doc_id,
            "filename": filename,
            "deletion_results": deletion_results
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
        
        filename = doc_info.get('filename')
        index_name = f"docs_{filename.replace('.', '_').replace(' ', '_')}"
        
        redis_client = get_redis_client()
        vector_exists = len(list(redis_client.scan_iter(match=f"doc:{index_name}:*", count=1))) > 0
        
        file_size = os.path.getsize(file_path) if file_exists else None
        
        return {
            **doc_info,
            "file_exists": file_exists,
            "vector_exists": vector_exists,
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
    start_time = time.time()
    
    try:
        embedding_model = get_embedding_model()
        redis_client = get_redis_client()
        
        # Get all index names
        all_keys = list(redis_client.scan_iter(match="doc:docs_*:0"))
        if not all_keys:
            return VectorSearchResponse(
                query=request.query,
                results=[],
                total_found=0,
                k_requested=request.k,
                similarity_threshold=request.similarity_threshold,
                search_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        index_names = set()
        for key in all_keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            parts = key_str.split(':')
            if len(parts) >= 2:
                index_names.add(parts[1])
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(request.query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        all_results = []
        
        # Search across all indexes
        for index_name in index_names:
            try:
                v = VectorQuery(
                    vector=query_vector.tolist(),
                    vector_field_name="embedding",
                    return_fields=["content", "doc_id", "filename", "uploaded_by", "created_at"],
                    num_results=request.k
                )
                
                schema = {
                    "index": {
                        "name": index_name,
                        "prefix": f"doc:{index_name}",
                        "storage_type": "hash"
                    },
                    "fields": [
                        {"name": "content", "type": "text"},
                        {"name": "doc_id", "type": "tag"},
                        {"name": "filename", "type": "text"},
                        {"name": "uploaded_by", "type": "text"},
                        {"name": "created_at", "type": "text"},
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
                index.connect(redis_client)
                
                results = index.query(v)
                
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
                                "similarity_score": similarity
                            }
                        })
                
            except Exception as e:
                logger.error(f"Error searching index {index_name}: {e}")
                continue
        
        # Sort and limit results
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
            search_time_ms=search_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/search-with-llm")
async def search_with_llm(
    request: VectorSearchRequest,
    current_user: dict = Depends(verify_token_v2)
):
    start_time = time.time()

    try:
        embedding_model = get_embedding_model()
        redis_client = get_redis_client()
        
        all_keys = list(redis_client.scan_iter(match="doc:docs_*:0"))
        if not all_keys:
            return {"llm_response": "No relevant information found in the provided documents."}
        
        index_names = set()
        for key in all_keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            parts = key_str.split(':')
            if len(parts) >= 2:
                index_names.add(parts[1])
        
        query_embedding = embedding_model.embed_query(request.query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        all_results = []
        
        for index_name in index_names:
            try:
                v = VectorQuery(
                    vector=query_vector.tolist(),
                    vector_field_name="embedding",
                    return_fields=["content", "doc_id", "filename"],
                    num_results=request.k
                )
                
                schema = {
                    "index": {"name": index_name, "prefix": f"doc:{index_name}", "storage_type": "hash"},
                    "fields": [
                        {"name": "content", "type": "text"},
                        {"name": "doc_id", "type": "tag"},
                        {"name": "filename", "type": "text"},
                        {"name": "embedding", "type": "vector", "attrs": {
                            "dims": len(query_embedding), "distance_metric": "cosine",
                            "algorithm": "flat", "datatype": "float32"}}
                    ]
                }
                
                index = SearchIndex.from_dict(schema)
                index.connect(redis_client)
                results = index.query(v)
                
                for result in results:
                    similarity = standardization(float(result.get('vector_distance', 1.0)))
                    if similarity >= request.similarity_threshold:
                        all_results.append({"content": result.get('content', ''), "score": similarity})
                
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:request.k]
        
        llm_response = "No relevant information found."
        if top_results:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                context = "\n\n".join([f"Doc {i+1}:\n{r['content']}" for i, r in enumerate(top_results)])
                
                prompt_template = PromptTemplate(
                    input_variables=["query", "context"],
                    template="""
🎯 Vai trò:
Bạn là một trợ lý AI chuyên nghiệp, chỉ trả lời dựa trên thông tin từ **tài liệu được cung cấp**.

📋 Nguyên tắc:
- Chỉ sử dụng thông tin từ tài liệu
- Không thêm kiến thức bên ngoài
- Không suy đoán hoặc giả định
- Nếu không có thông tin: "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."

📝 Cấu trúc trả lời:
1. **Câu mở đầu**: Tóm tắt ngắn gọn (1-2 câu)
2. **Nội dung chính**: Trình bày bằng danh sách có số thứ tự hoặc gạch đầu dòng
3. **Kết luận** (nếu cần): Tóm lược hoặc lời khuyên

💡 Format markdown:
- Dùng **số thứ tự** (1., 2., 3.) cho các bước hoặc quy trình
- Dùng **gạch đầu dòng** (-, *, •) cho danh sách các ý
- Dùng **bold** cho từ khóa quan trọng
- Dùng > cho trích dẫn từ tài liệu (nếu cần)

❓ Câu hỏi của người dùng:
{query}

📂 Tài liệu tham khảo:
{context}

Hãy trả lời câu hỏi dựa trên tài liệu trên.
"""
                )
                llm_response = llm.invoke(prompt_template.format(query=request.query, context=context)).content
            except Exception as e:
                logger.error(f"LLM failed: {e}")
        
        return {"llm_response": llm_response}
        
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

class ProcessQueryRequest(BaseModel):
    query: str
    thread_id: str | None = None

class ProcessQueryResponse(BaseModel):
    status: str
    data: dict | None
    error: str | None
    thread_id: str | None

@router.post("/process-query", response_model=ProcessQueryResponse)  
async def process_query_api(request: ProcessQueryRequest):
    start_time = time.time()
    try:
        result_json = process_query(request.query, thread_id=request.thread_id)
        result_dict = json.loads(result_json)
        
        try:
            validated_response = ProcessQueryResponse(**result_dict)
            return validated_response
        except ValueError as ve:
            logger.warning(f"Pydantic validation failed: {str(ve)}. Fallback to raw JSON.")
            return JSONResponse(content=result_dict, status_code=200, media_type="application/json")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi xử lý query: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Xử lý query mất {elapsed:.2f}s")