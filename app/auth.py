"""
Authentication dependencies for FastAPI
Handles JWT token verification and user authentication
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from database.config import get_db
from database.crud import get_user_by_id
from database.models import User
from utils.helpers import decode_access_token
from schemas.health_schema import TokenData
import logging

logger = logging.getLogger(__name__)

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user
    
    Usage in routes:
        @router.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.id}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode token
        payload = decode_access_token(token)
        if payload is None:
            logger.warning("Invalid token received")
            raise credentials_exception
        
        # Extract user information
        user_id: int = payload.get("user_id")
        username: str = payload.get("sub")
        
        if username is None or user_id is None:
            logger.warning("Token missing required claims")
            raise credentials_exception
        
        token_data = TokenData(user_id=user_id, username=username)
    
    except JWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise credentials_exception
    
    # Get user from database
    user = get_user_by_id(db, user_id=token_data.user_id)
    if user is None:
        logger.warning(f"User not found: {token_data.user_id}")
        raise credentials_exception
    
    if not user.is_active:
        logger.warning(f"Inactive user attempted access: {token_data.user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure user is active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to require admin privileges
    (Add admin field to User model if needed)
    """
    # This is a placeholder - add admin logic as needed
    return current_user
