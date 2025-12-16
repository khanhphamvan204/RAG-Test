from fastapi import APIRouter, HTTPException, Depends
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import Config  # Đảm bảo import này đúng
import os
import json
from app.services.auth_service import verify_token_v2
from bson import ObjectId
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

# Sử dụng connection pooling
_mongo_client = None

def get_mongo_client():
    """Get MongoDB client with connection pooling"""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                Config.DATABASE_URL,
                maxPoolSize=10,
                minPoolSize=2,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000
            )
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("MongoDB connection established in documents route")
        except Exception as e:
            logger.error(f"Failed to establish MongoDB connection: {e}")
            _mongo_client = None
            raise
    return _mongo_client

@router.get("/types", response_model=dict)
async def get_file_types(current_user: dict = Depends(verify_token_v2)):
    """Lấy danh sách các loại file được hỗ trợ."""
    return {
        "file_types": [
            {"value": "public", "label": "Thông báo chung (Public)", "description": "Tài liệu công khai cho tất cả người dùng"},
            {"value": "student", "label": "Sinh viên (Student)", "description": "Tài liệu dành cho sinh viên"},
            {"value": "teacher", "label": "Giảng viên (Teacher)", "description": "Tài liệu dành cho giảng viên"},
            {"value": "admin", "label": "Quản trị viên (Admin)", "description": "Tài liệu dành cho quản trị viên"}
        ]
    }

@router.get("/list", response_model=dict)
async def list_documents(
    file_type: str = None, 
    q: str = None,          
    limit: int = 100, 
    skip: int = 0,
    current_user: dict = Depends(verify_token_v2)
):
    """Lấy danh sách tài liệu (có phân trang & tìm kiếm) - lọc theo unit của user."""
    try:
        # Lấy thông tin user từ token
        unit_name = current_user.get('unit_name', 'default_unit')
        user_role = current_user.get('role', 'student')
        user_id = current_user.get('id')
        
        logger.info(f"[LIST] User {user_id} ({user_role}) from unit '{unit_name}' listing documents")
        
        documents = []
        total = 0

        # Thử lấy từ MongoDB
        try:
            client = get_mongo_client()
            db = client["faiss_db"]
            collection = db["metadata"]

            filter_dict = {}
            
            # Lọc theo unit_name (tất cả user chỉ xem tài liệu trong unit của mình)
            filter_dict["unit_name"] = unit_name
            
            if file_type:
                filter_dict["file_type"] = file_type
            if q:
                filter_dict["$or"] = [
                    {"filename": {"$regex": q, "$options": "i"}},
                    {"uploaded_by": {"$regex": q, "$options": "i"}},
                ]

            # Tính tổng
            total = collection.count_documents(filter_dict)

            # Lấy dữ liệu phân trang
            documents = list(
                collection.find(filter_dict)
                .skip(skip)
                .limit(limit)
                .sort("createdAt", -1)
            )

            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            return {
                "documents": documents,
                "total": total,
                "source": "mongodb",
                "showing": len(documents),
                "unit_name": unit_name,
                "user_role": user_role
            }
        
        except PyMongoError as e:
            logger.error(f"Failed to retrieve documents from MongoDB: {str(e)}")
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")

        # Fallback: JSON
        logger.info("Falling back to JSON files")
        
        # Xây dựng đường dẫn đến metadata.json dựa trên Config
        metadata_file = os.path.join(
            Config.DATA_PATH, 
            Config.FILE_PATHS.get('vector_folder', 'Rag_Info/Faiss_Folder'), 
            "metadata.json"
        )
        
        all_documents = []
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)

                # Lọc theo unit_name (tất cả user chỉ xem tài liệu trong unit của mình)
                metadata_list = [
                    item for item in metadata_list 
                    if item.get("unit_name", "default_unit") == unit_name
                ]
                
                # Áp dụng các filter khác
                if file_type:
                    metadata_list = [item for item in metadata_list if item.get("file_type") == file_type]
                if q:
                    metadata_list = [
                        item for item in metadata_list
                        if q.lower() in item.get("filename", "").lower()
                        or q.lower() in item.get("uploaded_by", "").lower()
                    ]

                all_documents.extend(metadata_list)
            else:
                logger.warning(f"JSON metadata file not found: {metadata_file}")
                
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {str(e)}")

        # Sort + paginate
        all_documents.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
        total = len(all_documents)
        documents = all_documents[skip: skip + limit]

        return {
            "documents": documents,
            "total": total,
            "source": "json",
            "showing": len(documents),
            "unit_name": unit_name,
            "user_role": user_role
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

def convert_to_str(d):
    """Recursively convert all values in a dictionary to strings."""
    if isinstance(d, dict):
        return {k: convert_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_str(i) for i in d]
    else:
        # Nếu là ObjectId, chuyển đổi sang str
        if isinstance(d, (ObjectId)):
            return str(d)
        return d

@router.get("/list/details/{document_id}", response_model=dict)
async def get_document_details(
    document_id: str,
    current_user: dict = Depends(verify_token_v2)
) -> Dict[str, Any]:
    """
    Lấy chi tiết tài liệu (chỉ 5 trường) – tối ưu cho frontend.
    """
    try:
        client = get_mongo_client()
        collection = client["faiss_db"]["metadata"]

        # Tìm bằng ObjectId hoặc string
        try:
            query = {"_id": ObjectId(document_id)}
        except Exception:
            query = {"_id": document_id}

        doc = collection.find_one(
            query,
            projection={
                "filename": 1,
                "url": 1,
                "uploaded_by": 1,
                "createdAt": 1,
            }
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Tài liệu không tồn tại")

        doc = convert_to_str(doc)  # _id, createdAt → str

        # Ẩn đường dẫn nội bộ
        if "url" in doc:
            doc["url"] = doc["url"].replace(Config.DATA_PATH, "/files", 1)

        return {"document": doc}

    except PyMongoError as e:
        logger.error(f"MongoDB error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi cơ sở dữ liệu")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống")

# Optional: Add cleanup function for app shutdown
def close_documents_mongo():
    """Close MongoDB connection for documents route"""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None