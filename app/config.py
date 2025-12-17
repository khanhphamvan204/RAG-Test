from dotenv import load_dotenv
import os
import re

load_dotenv()

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "Root_Folder")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
    # LLM Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # Options: gemini, openai, groq
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    FILE_PATHS = {
        'file_folder': os.getenv("FILE_FOLDER", "Rag_Info/File_Folder"),
        'vector_folder': os.getenv("VECTOR_FOLDER", "Rag_Info/Faiss_Folder")
    }
    
    @staticmethod
    def normalize_unit_name(unit_name: str) -> str:
        """
        Chuẩn hóa tên đơn vị thành folder name hợp lệ
        VD: "Phòng Đào tạo" -> "phong_dao_tao"
        """
        if not unit_name:
            return "default_unit"
        
        # Chuyển về lowercase và loại bỏ dấu tiếng Việt
        unit_name = unit_name.lower()
        
        # Map các ký tự có dấu sang không dấu
        vietnamese_map = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
        
        for viet, latin in vietnamese_map.items():
            unit_name = unit_name.replace(viet, latin)
        
        # Thay thế khoảng trắng và ký tự đặc biệt bằng underscore
        unit_name = re.sub(r'[^a-z0-9]+', '_', unit_name)
        
        # Loại bỏ underscore ở đầu và cuối
        unit_name = unit_name.strip('_')
        
        return unit_name if unit_name else "default_unit"
    
    @staticmethod
    def get_unit_file_folder(unit_name: str) -> str:
        """
        Lấy đường dẫn thư mục file cho đơn vị cụ thể
        VD: "Rag_Info/File_Folder/phong_dao_tao"
        """
        normalized = Config.normalize_unit_name(unit_name)
        base_path = Config.FILE_PATHS['file_folder']
        return os.path.join(base_path, normalized)
    
    @staticmethod
    def get_unit_vector_folder(unit_name: str) -> str:
        """
        Lấy đường dẫn thư mục vector cho đơn vị cụ thể
        VD: "Rag_Info/Faiss_Folder/phong_dao_tao"
        """
        normalized = Config.normalize_unit_name(unit_name)
        base_path = Config.FILE_PATHS['vector_folder']
        return os.path.join(base_path, normalized)
    
    @staticmethod
    def get_unit_index_name(unit_name: str) -> str:
        """
        Lấy tên Redis index cho đơn vị cụ thể
        VD: "unit_index_phong_dao_tao"
        """
        normalized = Config.normalize_unit_name(unit_name)
        return f"unit_index_{normalized}"