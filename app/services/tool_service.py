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
from app.services.embedding_service import get_embedding_model, get_redis_client, get_redis_url

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
        try:
            # Get embedding model
            embedding_model = get_embedding_model()
            redis_client = get_redis_client()
            redis_url = get_redis_url()  # Get Redis URL for SearchIndex
            
            # Get all index names
            all_keys = list(redis_client.scan_iter(match="doc:docs_*:0"))
            if not all_keys:
                logger.warning("No documents found in Redis")
                return RAGResponse(llm_response="Xin lỗi, tôi không tìm thấy thông tin này.", search_type="rag")
            
            # Extract unique index names
            index_names = set()
            for key in all_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                parts = key_str.split(':')
                if len(parts) >= 2:
                    index_names.add(parts[1])
            
            logger.info(f"Found {len(index_names)} indexes: {index_names}")
            
            # Generate query embedding
            query_embedding = embedding_model.embed_query(request.query)
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            all_results = []
            
            # Search across all indexes
            for index_name in index_names:
                try:
                    # Create VectorQuery
                    v = VectorQuery(
                        vector=query_vector.tolist(),
                        vector_field_name="embedding",
                        return_fields=["content", "doc_id", "filename", "uploaded_by", "created_at"],
                        num_results=request.k
                    )
                    
                    # Get index
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
                    index.connect(redis_url)  # Use Redis URL, not client object
                    
                    # Execute search
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
            
            # Sort by similarity and take top k
            all_results.sort(key=lambda x: x['metadata']['similarity_score'], reverse=True)
            top_results = all_results[:request.k]
            
            logger.info(f"Found {len(top_results)} results after filtering")
            
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
                        [f"Document {i+1}:\n{result['content']}" for i, result in enumerate(top_results)]
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
                    
                except Exception as e:
                    logger.error(f"LLM response generation failed: {str(e)}")
                    llm_response = "Không thể tạo câu trả lời từ LLM."
            
            return RAGResponse(llm_response=llm_response, search_type="rag")
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return RAGResponse(llm_response="Lỗi hệ thống.", search_type="rag")

# Initialize service
rag_service = RAGSearchService()

# Define the RAG search tool for LangGraph
rag_search_tool = StructuredTool.from_function(
    func=lambda request: rag_service.search_with_llm(VectorSearchRequest(**request)) if isinstance(request, dict) else rag_service.search_with_llm(request),
    name="vector_rag_search",
    description="Thực hiện RAG search trên Redis vector database để tìm tài liệu tương tự và generate câu trả lời từ LLM (Gemini). Input là query văn bản, k (top results), similarity_threshold. Trả về llm_response với câu trả lời dựa trên context từ tài liệu."
)