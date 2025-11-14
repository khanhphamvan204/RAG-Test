from pydantic import BaseModel
import os
import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
import sys
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.models.vector_models import VectorSearchRequest
from app.services.embedding_service import (
    get_embedding_model, 
    get_redis_client, 
    get_redis_url,
    UNIFIED_INDEX_NAME
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardization(distance: float) -> float:
    """Chuyển đổi cosine distance thành similarity score"""
    return 1 - distance

class RAGResponse(BaseModel):
    llm_response: str
    search_type: str = "rag"

class RAGSearchService:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                temperature=0.1,
            )
        else:
            self.llm = None

    def search_with_llm(self, request: VectorSearchRequest) -> RAGResponse:
        """
        Search trong unified index
        """
        try:
            embedding_model = get_embedding_model()
            redis_client = get_redis_client()
            redis_url = get_redis_url()
            
            # Kiểm tra xem có documents trong unified index không
            pattern = f"doc:{UNIFIED_INDEX_NAME}:*"
            sample_keys = list(redis_client.scan_iter(match=pattern, count=1))
            if not sample_keys:
                logger.warning("Không tìm thấy documents trong unified index")
                return RAGResponse(
                    llm_response="Xin lỗi, tôi không tìm thấy thông tin này.", 
                    search_type="rag"
                )
            
            logger.info(f"Tìm kiếm trong unified index: {UNIFIED_INDEX_NAME}")
            
            # Generate query embedding
            query_embedding = embedding_model.embed_query(request.query)
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            # Tạo VectorQuery cho unified index
            v = VectorQuery(
                vector=query_vector.tolist(),
                vector_field_name="embedding",
                return_fields=["content", "doc_id", "filename", "uploaded_by", "created_at", "chunk_id"],
                num_results=request.k * 2
            )
            
            # Schema cho unified index
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
                            "similarity_score": similarity
                        }
                    })
            
            # Sort và take top k
            all_results.sort(key=lambda x: x['metadata']['similarity_score'], reverse=True)
            top_results = all_results[:request.k]
            
            logger.info(f"Tìm thấy {len(top_results)} kết quả sau khi lọc (ngưỡng: {request.similarity_threshold})")
            
            # Generate LLM response
            llm_response = "Xin lỗi, tôi không tìm thấy thông tin này."
            if top_results:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=os.getenv('GOOGLE_API_KEY'),
                        temperature=0.3
                    )
                    
                    context = "\n\n".join(
                        [f"Tài liệu {i+1} (từ {result['metadata']['filename']}):\n{result['content']}" 
                         for i, result in enumerate(top_results)]
                    )
                    
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
                    
                    prompt = prompt_template.format(query=request.query, context=context)
                    llm_response = llm.invoke(prompt).content
                    
                    logger.info(f"Đã tạo LLM response thành công (sử dụng {len(top_results)} documents)")
                    
                except Exception as e:
                    logger.error(f"Lỗi tạo LLM response: {str(e)}")
                    llm_response = "Không thể tạo câu trả lời từ LLM."
            
            return RAGResponse(llm_response=llm_response, search_type="rag")
            
        except Exception as e:
            logger.error(f"Lỗi không mong đợi trong RAG search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return RAGResponse(llm_response="Lỗi hệ thống.", search_type="rag")


# Initialize service
rag_service = RAGSearchService()


# WRAPPER FUNCTION CHO RAG

def rag_search_wrapper(query, k=5, similarity_threshold=0.5):
    """Wrapper trả về dict cho RAG search"""
    logger.info(f"[WRAPPER] rag_search được gọi với query: {query[:50]}...")
    
    result = rag_service.search_with_llm(
        VectorSearchRequest(
            query=query,
            k=k,
            similarity_threshold=similarity_threshold
        )
    )
    
    # Trả về dict structured
    output = {
        "llm_response": result.llm_response,
        "source": "rag",
        "search_type": "rag",
        "activities_raw": [],  # RAG không có activities
        "total": 0
    }
    
    logger.info("[WRAPPER] Trả về RAG response (không có activities)")
    
    return output


# LANGCHAIN TOOL - SỬ DỤNG WRAPPER

rag_search_tool = StructuredTool.from_function(
    func=rag_search_wrapper,
    name="vector_rag_search",
    description=f"""
Thực hiện RAG search trên unified Redis vector database ({UNIFIED_INDEX_NAME}) để tìm tài liệu tương tự và generate câu trả lời từ LLM (Gemini).

Input parameters:
- query (str): Câu hỏi của người dùng
- k (int, default=5): Số lượng documents cần lấy
- similarity_threshold (float, default=0.5): Ngưỡng similarity tối thiểu (0-1)

Output: 
- llm_response: Câu trả lời được generate từ LLM
- source: "rag"
- search_type: "rag"
- activities_raw: [] (RAG không trả về hoạt động)

Index hiện tại: {UNIFIED_INDEX_NAME} (chứa tất cả documents của hệ thống)
"""
)