# app/auth/jwt_auth.py
import time
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


# --- Cấu hình ---
logger = logging.getLogger(__name__)
security = HTTPBearer()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") 
JWT_ALGORITHM = "HS256"

if not JWT_SECRET_KEY:
    logger.error("FATAL: JWT_SECRET_KEY is not set in environment variables!")

# --- Hàm 1: Kiểm tra token "Hợp lý" (v2) ---
def verify_token_v2(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Xác thực token "hợp lý" (v2):
    1. Check chữ ký (signature) dùng JWT_SECRET_KEY.
    2. Check các claim chuẩn (exp, nbf, iat).
    3. Check cấu trúc payload (phải có 'sub' và 'role').
    
    Nếu thành công, trả về payload.
    Nếu thất bại, raise HTTPException 401 (Unauthorized).
    """
    try:
        token = credentials.credentials
        
        # Bước 1 & 2: Decode token.
        # pyjwt sẽ tự động kiểm tra 'exp', 'nbf', 'iat' và chữ ký
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        
        # Bước 3: Kiểm tra cấu trúc payload
        # Cả 'sub' (user_id) và 'role' đều phải tồn tại và không được rỗng
        required_fields = ["sub", "role"]
        
        for field in required_fields:
            if field not in payload or not payload.get(field):
                raise jwt.InvalidTokenError(f"Token is missing required claim: '{field}'")
        
        # Token hợp lệ và đúng cấu trúc
        return payload
    
    except jwt.ExpiredSignatureError:
        # Lỗi khi token đã hết hạn
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        # Lỗi khi token không hợp lệ (sai chữ ký, sai cấu trúc,...)
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Các lỗi không mong muốn khác
        raise HTTPException(
            status_code=401,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Hàm 2: Kiểm tra vai trò "Advisor" ---
def verify_advisor_role(payload: Dict[str, Any] = Depends(verify_token_v2)) -> Dict[str, Any]:
    """
    Xác thực người dùng phải là 'advisor'.
    
    Hàm này "phụ thuộc" (Depends on) vào `verify_token_v2`.
    Nó chỉ chạy KHI `verify_token_v2` đã chạy thành công.
    
    Nếu vai trò không đúng, raise HTTPException 403 (Forbidden).
    """
    
    user_role = payload.get("role")
    
    if user_role != "advisor":
        logger.warning(f"Access denied: User (sub: {payload.get('sub')}) with role '{user_role}' tried to access advisor-only route.")
        raise HTTPException(
            status_code=403, 
            detail="Access denied: This resource is for advisors only."
        )
    
    return payload