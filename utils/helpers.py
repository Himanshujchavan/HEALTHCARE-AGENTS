from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from utils.constants import ALGORITHM, SECRET_KEY


_password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a submitted password against its stored bcrypt hash."""
    return _password_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Create the bcrypt hash stored for a user password."""
    return _password_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT containing the supplied claims."""
    claims = dict(data)
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=15)
    )
    claims["exp"] = expire
    return jwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode a JWT, returning None for invalid or expired credentials."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def normalize_level(level: str) -> str:
    """
    Unifies risk levels across different agents.
    Maps:
    - MEDIUM -> Moderate
    - LOW -> Low
    - HIGH -> High
    - CRITICAL -> Critical
    - Very Low -> Low
    - Very High -> Critical
    """
    if not level:
        return "Unknown"

    mapping = {
        "MEDIUM": "Moderate",
        "Moderate": "Moderate",
        "LOW": "Low",
        "Low": "Low",
        "Very Low": "Low",
        "HIGH": "High",
        "High": "High",
        "CRITICAL": "Critical",
        "Critical": "Critical",
        "Very High": "Critical",
    }
    return mapping.get(level, level)
