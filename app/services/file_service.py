import os
import logging
from fastapi import HTTPException
from app.config import Config

logger = logging.getLogger(__name__)

def get_file_paths(filename: str, unit_name: str = "default_unit") -> tuple[str, str]:
    """
    Lấy đường dẫn file và vector database dựa trên unit_name
    
    Args:
        filename: Tên file
        unit_name: Tên đơn vị từ token (VD: "Phòng Đào tạo")
    
    Returns:
        tuple: (file_path, vector_db_path)
    
    Raises:
        HTTPException: Nếu filename rỗng hoặc có lỗi config
    """
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    try:
        # Lấy thư mục unit-specific
        file_folder = Config.get_unit_file_folder(unit_name)
        vector_folder = Config.get_unit_vector_folder(unit_name)
        
        # Tạo đường dẫn đầy đủ với base path
        base_path = Config.DATA_PATH
        file_path = os.path.join(base_path, file_folder, filename).replace("\\", "/")
        vector_db_path = os.path.join(base_path, vector_folder).replace("\\", "/")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        
        logger.info(f"get_file_paths called: filename={filename}, unit={unit_name}")
        logger.info(f"Returning: file_path={file_path}, vector_db_path={vector_db_path}")
        
        return file_path, vector_db_path
        
    except KeyError as e:
        logger.error(f"Config error: Missing key {str(e)} in FILE_PATHS")
        raise HTTPException(
            status_code=500, 
            detail=f"Configuration error: Missing {str(e)} in FILE_PATHS"
        )
    except Exception as e:
        logger.error(f"Error in get_file_paths: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting file paths: {str(e)}"
        )

def get_unit_folder_info(unit_name: str) -> dict:
    """
    Lấy thông tin về thư mục của unit
    """
    file_folder = Config.get_unit_file_folder(unit_name)
    vector_folder = Config.get_unit_vector_folder(unit_name)
    normalized_name = Config.normalize_unit_name(unit_name)
    
    return {
        "unit_name": unit_name,
        "normalized_name": normalized_name,
        "file_folder": file_folder,
        "vector_folder": vector_folder,
        "file_folder_exists": os.path.exists(file_folder),
        "vector_folder_exists": os.path.exists(vector_folder),
        "index_name": Config.get_unit_index_name(unit_name)
    }

def list_unit_files(unit_name: str) -> list:
    """
    Liệt kê tất cả files trong thư mục của unit
    """
    file_folder = Config.get_unit_file_folder(unit_name)
    
    if not os.path.exists(file_folder):
        return []
    
    files = []
    for filename in os.listdir(file_folder):
        file_path = os.path.join(file_folder, filename)
        if os.path.isfile(file_path):
            files.append({
                "filename": filename,
                "path": file_path,
                "size": os.path.getsize(file_path)
            })
    
    return files