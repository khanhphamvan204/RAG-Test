from pydantic import BaseModel
import os
import logging
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
import numpy as np

from app.models.vector_models import VectorSearchRequest
from app.services.embedding_service import (
    get_embedding_model, 
    get_redis_client, 
    get_redis_url
)
from app.config import Config

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardization(distance: float) -> float:
    """Chuyển đổi cosine distance thành similarity score"""
    return 1 - distance

class RAGResponse(BaseModel):
    llm_response: str
    search_type: str = "rag"
    unit_name: str = ""
    context: Optional[List[dict]] = None  # Danh sách tài liệu được retrieval

class RAGSearchService:
    def __init__(self):
        provider = Config.LLM_PROVIDER.lower()
        logger.info(f"[RAG_SERVICE] Initializing with LLM provider: {provider}")
        
        try:
            if provider == "groq":
                api_key = Config.GROQ_API_KEY
                if not api_key:
                    logger.warning("[RAG_SERVICE] GROQ_API_KEY not set, LLM disabled")
                    self.llm = None
                else:
                    model_name = Config.GROQ_MODEL
                    logger.info(f"[RAG_SERVICE] Using Groq model: {model_name}")
                    self.llm = ChatGroq(
                        model=model_name,
                        groq_api_key=api_key,
                        temperature=0.1,
                    )
            elif provider == "openai":
                api_key = Config.OPENAI_API_KEY
                if not api_key:
                    logger.warning("[RAG_SERVICE] OPENAI_API_KEY not set, LLM disabled")
                    self.llm = None
                else:
                    model_name = Config.OPENAI_MODEL
                    logger.info(f"[RAG_SERVICE] Using OpenAI model: {model_name}")
                    self.llm = ChatOpenAI(
                        model=model_name,
                        openai_api_key=api_key,
                        temperature=0.1,
                    )
            else:
                logger.error(f"[RAG_SERVICE] Invalid LLM_PROVIDER: {provider}")
                self.llm = None
        except Exception as e:
            logger.error(f"[RAG_SERVICE] Error initializing LLM: {e}")
            self.llm = None

    def search_with_llm(self, request: VectorSearchRequest, unit_name: str = "default_unit") -> RAGResponse:
        """
        Search trong unit-specific index
        """
        try:
            embedding_model = get_embedding_model()
            redis_client = get_redis_client()
            redis_url = get_redis_url()
            
            # Lấy index name của unit
            index_name = Config.get_unit_index_name(unit_name)
            
            logger.info(f"[RAG] Searching in unit index: {index_name}")
            
            # Kiểm tra documents trong unit index
            pattern = f"doc:{index_name}:*"
            sample_keys = list(redis_client.scan_iter(match=pattern, count=1))
            
            if not sample_keys:
                logger.warning(f"No documents found in unit index: {index_name}")
                return RAGResponse(
                    llm_response=f"Xin lỗi, không tìm thấy tài liệu nào trong thư viện của đơn vị '{unit_name}'.", 
                    search_type="rag",
                    unit_name=unit_name
                )
            
            # Generate query embedding
            query_embedding = embedding_model.embed_query(request.query)
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            # Vector query - chỉ lấy đúng k results để tối ưu tokens
            v = VectorQuery(
                vector=query_vector.tolist(),
                vector_field_name="embedding",
                return_fields=["content", "doc_id", "filename", "uploaded_by", "created_at", "chunk_id", "unit_name"],
                num_results=min(request.k, 3)  # Giới hạn tối đa 3 documents để tránh 429 TPM
            )
            
            # Schema cho unit index
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
            
            # Connect và search
            index = SearchIndex.from_dict(schema)
            index.connect(redis_url)
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
            
            # Sort và take top k
            all_results.sort(key=lambda x: x['metadata']['similarity_score'], reverse=True)
            top_results = all_results[:request.k]
            
            logger.info(f"[RAG] Found {len(top_results)} results in unit '{unit_name}'")
            
            # Generate LLM response
            llm_response = f"Xin lỗi, không tìm thấy thông tin liên quan trong thư viện của đơn vị '{unit_name}'."
            
            if top_results:
                try:
                    provider = Config.LLM_PROVIDER.lower()
                    
                    if provider == "groq":
                        llm = ChatGroq(
                            model=Config.GROQ_MODEL,
                            groq_api_key=Config.GROQ_API_KEY,
                            temperature=0.3
                        )
                    elif provider == "openai":
                        llm = ChatOpenAI(
                            model=Config.OPENAI_MODEL,
                            openai_api_key=Config.OPENAI_API_KEY,
                            temperature=0.3
                        )
                    else:
                        raise ValueError(f"Invalid LLM_PROVIDER: {provider}")
                    
                    context = "\n\n".join(
                        [f"Tài liệu {i+1} (từ {result['metadata']['filename']}):\n{result['content']}" 
                         for i, result in enumerate(top_results)]
                    )
                    
                    # Log context để debug
                    logger.info(f"[RAG] Context length: {len(context)} characters")
                    logger.debug(f"[RAG] Context preview: {context[:500]}...")
                    
                    prompt_template = PromptTemplate(
                        input_variables=["query", "context", "unit_name"],
                        template="""
Vai trò:
Bạn là trợ lý AI của thư viện đơn vị "{unit_name}", chỉ trả lời dựa trên tài liệu được cung cấp.

Nguyên tắc:
- Chỉ sử dụng thông tin từ tài liệu của đơn vị "{unit_name}"
- Không thêm kiến thức bên ngoài
- Không suy đoán hoặc giả định
- Nếu không có thông tin: "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."

Cấu trúc trả lời:
1. **Câu mở đầu**: Tóm tắt ngắn gọn (1-2 câu)
2. **Nội dung chính**: Trình bày bằng danh sách có số thứ tự hoặc gạch đầu dòng
3. **Kết luận** (nếu cần): Tóm lược hoặc lời khuyên

Format markdown:
- Dùng **số thứ tự** (1., 2., 3.) cho các bước hoặc quy trình
- Dùng **gạch đầu dòng** (-, *, •) cho danh sách các ý
- Dùng **bold** cho từ khóa quan trọng

Câu hỏi:
{query}

Tài liệu từ thư viện đơn vị "{unit_name}":
{context}

Hãy trả lời dựa trên tài liệu trên.
"""
                    )
                    
                    prompt = prompt_template.format(
                        query=request.query, 
                        context=context,
                        unit_name=unit_name
                    )
                    llm_response = llm.invoke(prompt).content
                    
                    logger.info(f"[RAG] Generated response using {len(top_results)} documents from unit '{unit_name}'")
                    
                except Exception as e:
                    logger.error(f"[RAG] LLM error: {str(e)}")
                    llm_response = "Không thể tạo câu trả lời từ LLM."
            
            # Chuẩn bị context để trả về
            retrieval_context = [
                {
                    "content": result['content'],
                    "filename": result['metadata'].get('filename', 'unknown'),
                    "similarity": result['metadata'].get('similarity_score', 0),
                    "doc_id": result['metadata'].get('doc_id', ''),
                    "chunk_id": result['metadata'].get('chunk_id', 0)
                }
                for result in top_results
            ] if top_results else None
            
            return RAGResponse(
                llm_response=llm_response, 
                search_type="rag",
                unit_name=unit_name,
                context=retrieval_context
            )
            
        except Exception as e:
            logger.error(f"[RAG] Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return RAGResponse(
                llm_response="Lỗi hệ thống.", 
                search_type="rag",
                unit_name=unit_name
            )


# Initialize service
rag_service = RAGSearchService()


# WRAPPER FUNCTION CHO RAG với unit support

def rag_search_wrapper(query: str, unit_name: Optional[str] = None, k: int = 3, similarity_threshold: float = 0.3):
    """
    Wrapper trả về dict cho RAG search với unit support
    
    Args:
        query: Câu hỏi
        unit_name: Tên đơn vị (nếu None thì lấy từ langgraph_service)
        k: Số lượng results
        similarity_threshold: Ngưỡng similarity
    """
    # Nếu không truyền unit_name, lấy từ langgraph_service
    if unit_name is None:
        try:
            from app.services.langgraph_service import get_current_unit_name
            unit_name = get_current_unit_name()
            logger.info(f"[RAG_WRAPPER] Got unit_name from context: {unit_name}")
        except Exception as e:
            logger.warning(f"[RAG_WRAPPER] Cannot get unit_name from context: {e}, using default")
            unit_name = "default_unit"
    
    # Debug: log query với encoding UTF-8
    try:
        logger.info(f"[RAG_WRAPPER] Query in unit '{unit_name}': {query[:50]}...")
        logger.debug(f"[RAG_WRAPPER] Query type: {type(query)}, length: {len(query)}")
        logger.debug(f"[RAG_WRAPPER] Query repr: {repr(query[:50])}")
    except Exception as e:
        logger.error(f"[RAG_WRAPPER] Error logging query: {e}")
    
    result = rag_service.search_with_llm(
        VectorSearchRequest(
            query=query,
            k=k,
            similarity_threshold=similarity_threshold
        ),
        unit_name=unit_name
    )
    
    # Trả về dict structured với context
    output = {
        "llm_response": result.llm_response,
        "source": "rag",
        "search_type": "rag",
        "unit_name": unit_name,
        "activities_raw": [],  # RAG không có activities
        "total": 0,
        "context": result.context  # Thêm context từ RAG
    }
    
    context_count = len(result.context) if result.context else 0
    logger.info(f"[RAG_WRAPPER] Response from unit '{unit_name}' with {context_count} documents")
    
    return output


# LANGCHAIN TOOL - với unit support

rag_search_tool = StructuredTool.from_function(
    func=rag_search_wrapper,
    name="vector_rag_search",
    description="""
Thực hiện RAG search trên Redis vector database theo ĐƠN VỊ (unit-based) để tìm tài liệu và generate câu trả lời.

**QUAN TRỌNG**: Mỗi đơn vị có thư viện riêng, chỉ search trong thư viện của đơn vị hiện tại.

**CÁCH SỬ DỤNG**:
- Nếu KHÔNG truyền unit_name: Tool sẽ tự động lấy unit_name từ context (người dùng hiện tại)
- Nếu CÓ truyền unit_name: Tool sẽ search trong unit được chỉ định (chỉ admin mới được dùng)

Input parameters:
- query (str, REQUIRED): Câu hỏi của người dùng
- unit_name (str, OPTIONAL): Tên đơn vị - NẾU KHÔNG TRUYỀN thì dùng unit của user hiện tại
- k (int, default=3): Số lượng documents
- similarity_threshold (float, default=0.3): Ngưỡng similarity (0-1)

Output: 
- llm_response: Câu trả lời từ LLM dựa trên tài liệu
- source: "rag"
- unit_name: Tên đơn vị đã search
- context: Danh sách tài liệu retrieval được
- activities_raw: [] (RAG không trả về hoạt động)

**LƯU Ý**: 
- Tool tự động lấy unit_name từ context, KHÔNG CẦN truyền thủ công
- Chỉ search trong thư viện của đơn vị được chỉ định
- Ví dụ gọi tool: vector_rag_search(query="quy định học vụ")
"""
)