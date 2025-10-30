from pydantic import BaseModel
import os
import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.models.vector_models import VectorSearchRequest
from app.services.embedding_service import get_embedding_model
from app.services.file_service import get_file_paths

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardization(distance: float) -> float:
    """Chuyển đổi khoảng cách L2 thành điểm tương đồng (similarity score) trong khoảng [0, 1]."""
    if distance < 0:
        return 0.0
    else:
        return 1 / (1 + distance)

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
            _, vector_db_path = get_file_paths("dummy_filename")

            # Check if vector DB exists
            if not (os.path.exists(f"{vector_db_path}/index.faiss") and os.path.exists(f"{vector_db_path}/index.pkl")):
                logger.warning(f"Vector DB not found at {vector_db_path}")
                return RAGResponse(llm_response="Xin lỗi, tôi không tìm thấy thông tin này.", search_type="rag")

            # Load vector DB
            try:
                embedding_model = get_embedding_model()
                db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
                logger.info(f"Vector DB loaded with {db.index.ntotal} documents")
            except Exception as e:
                logger.error(f"Failed to load vector database: {str(e)}")
                return RAGResponse(llm_response=f"Không thể tải vector database: {str(e)}", search_type="rag")

            # Perform similarity search
            try:
                docs_with_scores = db.similarity_search_with_score(
                    request.query,
                    k=request.k
                )
                logger.info(f"Raw search: {len(docs_with_scores)} docs, scores: {[score for _, score in docs_with_scores]}")

                # Chuẩn hóa và lọc theo threshold
                filtered_docs = [
                    (doc, standardization(score)) for doc, score in docs_with_scores
                ]

                # Chuyển sang dict
                search_results = [
                    {
                        "content": doc.page_content,
                        "metadata": {**doc.metadata, "similarity_score": float(score)}
                    }
                    for doc, score in filtered_docs
                ]

                # Lấy top k
                top_results = search_results[:request.k]

                # Generate LLM response
                llm_response = "Xin lỗi, tôi không tìm thấy thông tin này."
                if top_results:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            google_api_key=os.getenv('GOOGLE_API_KEY'),
                            temperature=0.3
                        )
                        if not llm:
                            llm_response = "LLM không được cấu hình."
                        else:
                            context = "\n\n".join(
                                [f"Document {i+1}:\n{result['content']}" for i, result in enumerate(top_results)]
                            )
                            logger.info(f"Context length: {len(context)} chars")

                            prompt_template = PromptTemplate(
                                input_variables=["query", "context"],
                                template="""
Bạn là trợ lý hữu ích trả lời dựa trên context được cung cấp. 
Nếu context có thông tin liên quan (dù ít), hãy trả lời ngắn gọn bằng tiếng Việt. 
Chỉ dùng câu "Xin lỗi, tôi không tìm thấy thông tin này." nếu context hoàn toàn không liên quan.

Cấu trúc:
- Tóm tắt ngắn gọn.
- Bullet points nếu cần.

Query: {query}

Context:
{context}
Answer:"""
                            )

                            prompt = prompt_template.format(query=request.query, context=context)
                            llm_response = llm.invoke(prompt).content

                    except Exception as e:
                        logger.error(f"LLM response generation failed: {str(e)}")
                        llm_response = "Không thể tạo câu trả lời từ LLM."

                return RAGResponse(llm_response=llm_response, search_type="rag")

            except Exception as e:
                logger.error(f"Search execution failed: {str(e)}")
                return RAGResponse(llm_response=f"Tìm kiếm thất bại: {str(e)}", search_type="rag")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return RAGResponse(llm_response="Lỗi hệ thống.", search_type="rag")

# Initialize service
rag_service = RAGSearchService()

# Define the RAG search tool for LangGraph
rag_search_tool = StructuredTool.from_function(
    func=lambda request: rag_service.search_with_llm(VectorSearchRequest(**request)) if isinstance(request, dict) else rag_service.search_with_llm(request),
    name="vector_rag_search",
    description="Thực hiện RAG search trên vector database (FAISS) để tìm tài liệu tương tự và generate câu trả lời từ LLM (Gemini). Input là query văn bản, k (top results), similarity_threshold. Trả về llm_response với câu trả lời dựa trên context từ tài liệu, hoặc thông báo lỗi nếu không có dữ liệu phù hợp."
)