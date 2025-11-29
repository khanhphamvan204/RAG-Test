from typing import Optional, List
from pydantic import BaseModel, Field

class AddVectorRequest(BaseModel):
    """Model cho request thêm vector document"""
    id: str = Field(alias="_id")
    filename: str
    url: str
    uploaded_by: str
    createdAt: str
    unit_name: Optional[str] = "default_unit"  # Tên đơn vị
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class VectorSearchRequest(BaseModel):
    """Model cho request tìm kiếm vector"""
    query: str = Field(..., description="Câu truy vấn tìm kiếm")
    k: int = Field(default=5, ge=1, le=100, description="Số lượng kết quả trả về (1-100)")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Ngưỡng độ tương đồng (0.0-1.0)")
    unit_name: Optional[str] = None  # Tên đơn vị (sẽ lấy từ token nếu không có)

class SearchResultMetadata(BaseModel):
    """Metadata của kết quả tìm kiếm"""
    doc_id: str
    filename: str
    uploaded_by: str
    created_at: str
    chunk_id: int
    similarity_score: float
    unit_name: Optional[str] = None  # Tên đơn vị

class SearchResult(BaseModel):
    """Model cho một kết quả tìm kiếm"""
    content: str
    metadata: SearchResultMetadata

class VectorSearchResponse(BaseModel):
    """Response cho vector search"""
    query: str
    results: List[SearchResult]
    total_found: int
    k_requested: int
    similarity_threshold: float
    search_time_ms: float
    index_name: Optional[str] = None
    unit_name: Optional[str] = None  # Tên đơn vị đã search

