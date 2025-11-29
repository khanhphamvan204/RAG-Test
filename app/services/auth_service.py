import time
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
security = HTTPBearer()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") 
JWT_ALGORITHM = "HS256"

if not JWT_SECRET_KEY:
    logger.error("FATAL: JWT_SECRET_KEY is not set in environment variables!")

def verify_token_v2(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Xác thực token và trả về payload + raw token + unit_name
    """
    try:
        token = credentials.credentials
        
        logger.info(f"[AUTH] Raw token received: {token[:30]}...")
        
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        
        required_fields = ["sub", "role"]
        for field in required_fields:
            if field not in payload or not payload.get(field):
                raise jwt.InvalidTokenError(f"Token is missing required claim: '{field}'")
        
        # LƯU TOKEN GỐC VÀO PAYLOAD
        payload['token'] = token
        payload['id'] = payload.get('sub')
        
        # TRÍCH XUẤT UNIT_NAME TỪ TOKEN
        unit_name = payload.get('unit_name', 'default_unit')
        payload['unit_name'] = unit_name
        
        logger.info(f"[AUTH] Token verified successfully")
        logger.info(f"[AUTH] User ID: {payload.get('id')}, Role: {payload.get('role')}, Unit: {unit_name}")
        logger.info(f"[AUTH] Payload keys: {list(payload.keys())}")
        
        return payload
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_advisor_role(payload: Dict[str, Any] = Depends(verify_token_v2)) -> Dict[str, Any]:
    """
    Xác thực người dùng phải là 'advisor'
    """
    user_role = payload.get("role")
    
    if user_role != "advisor":
        logger.warning(f"Access denied: User (sub: {payload.get('sub')}) with role '{user_role}' tried to access advisor-only route.")
        raise HTTPException(
            status_code=403, 
            detail="Access denied: This resource is for advisors only."
        )
    
    return payload

def verify_admin_or_advisor_role(payload: Dict[str, Any] = Depends(verify_token_v2)) -> Dict[str, Any]:
    """
    Xác thực người dùng phải là 'admin' hoặc 'advisor'
    (Những người có quyền quản lý tài liệu)
    """
    user_role = payload.get("role")
    
    if user_role not in ["admin", "advisor"]:
        logger.warning(f"Access denied: User (sub: {payload.get('sub')}) with role '{user_role}' tried to access admin/advisor-only route.")
        raise HTTPException(
            status_code=403, 
            detail="Access denied: This resource requires admin or advisor role."
        )
    
    return payload