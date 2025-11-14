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
    Xac thuc token va tra ve payload + raw token
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
        
        # LUU TOKEN GOC VAO PAYLOAD
        payload['token'] = token
        payload['id'] = payload.get('sub')
        
        logger.info(f"[AUTH] Token verified successfully")
        logger.info(f"[AUTH] Payload keys: {list(payload.keys())}")
        logger.info(f"[AUTH] Token stored in payload: {payload.get('token')[:30]}...")
        
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
    Xac thuc nguoi dung phai la 'advisor'
    """
    user_role = payload.get("role")
    
    if user_role != "advisor":
        logger.warning(f"Access denied: User (sub: {payload.get('sub')}) with role '{user_role}' tried to access advisor-only route.")
        raise HTTPException(
            status_code=403, 
            detail="Access denied: This resource is for advisors only."
        )
    
    return payload