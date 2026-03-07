"""
Utility functions for logging, authentication, and common operations
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from utils.constants import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def setup_logging():
    """
    Configure logging for the application
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password (truncate to 72 characters for bcrypt compatibility)"""
    # Bcrypt has a 72 byte limit - truncate if needed
    return pwd_context.hash(password[:72])


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode JWT access token
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def format_timestamp(dt: datetime) -> str:
    """Format datetime for API responses"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def calculate_risk_score(abnormal_count: int, parameter_values: dict) -> int:
    """
    Calculate overall risk score based on abnormal parameters
    Returns: 0 (Very Low) to 4 (Very High)
    """
    if abnormal_count == 0:
        return 0  # Very Low
    elif abnormal_count == 1:
        return 1  # Low
    elif abnormal_count == 2:
        return 2  # Moderate
    elif abnormal_count == 3:
        return 3  # High
    else:
        return 4  # Very High
