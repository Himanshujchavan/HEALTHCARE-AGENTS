"""
CRUD operations for database models
Separates database logic from route handlers
"""
from sqlalchemy.orm import Session
from database.models import User, HealthRecord
from schemas.health_schema import HealthInput
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ==================== USER CRUD ====================

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, username: str, email: str, hashed_password: str) -> User:
    """Create new user"""
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User created: {username} (ID: {db_user.id})")
    return db_user


# ==================== HEALTH RECORD CRUD ====================

def create_health_record(db: Session, user_id: int, data: HealthInput) -> HealthRecord:
    """
    Create a new health record for a user
    """
    record = HealthRecord(
        user_id=user_id,
        hba1c=data.hba1c,
        glucose=data.glucose,
        bmi=data.bmi,
        age=data.age
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    logger.info(f"Health record created: ID={record.id}, User={user_id}")
    return record


def get_health_record(db: Session, record_id: int) -> Optional[HealthRecord]:
    """Get health record by ID"""
    return db.query(HealthRecord).filter(HealthRecord.id == record_id).first()


def get_user_health_records(
    db: Session, 
    user_id: int, 
    skip: int = 0, 
    limit: int = 100
) -> List[HealthRecord]:
    """Get all health records for a user with pagination"""
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == user_id)
        .order_by(HealthRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_health_record_analysis(
    db: Session, 
    record_id: int, 
    analysis_result: str
) -> Optional[HealthRecord]:
    """
    Update health record with analysis result
    """
    record = db.query(HealthRecord).filter(HealthRecord.id == record_id).first()
    if record:
        record.analyzed = True
        record.analysis_result = analysis_result
        db.commit()
        db.refresh(record)
        logger.info(f"Health record analyzed: ID={record_id}")
    return record


def delete_health_record(db: Session, record_id: int) -> bool:
    """Delete a health record"""
    record = db.query(HealthRecord).filter(HealthRecord.id == record_id).first()
    if record:
        db.delete(record)
        db.commit()
        logger.info(f"Health record deleted: ID={record_id}")
        return True
    return False


def get_latest_health_record(db: Session, user_id: int) -> Optional[HealthRecord]:
    """Get the most recent health record for a user"""
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == user_id)
        .order_by(HealthRecord.created_at.desc())
        .first()
    )
