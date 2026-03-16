"""
CRUD operations for database models (auth-free version)
Handles health records without requiring authentication
"""
from sqlalchemy.orm import Session
from database.models import HealthRecord
from schemas.health_schema import HealthInput
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# ==================== DUMMY USER ====================
# Use this constant for auth-free testing
USER_ID = 1


# ==================== HEALTH RECORD CRUD ====================

def create_health_record(db: Session, data: HealthInput) -> HealthRecord:
    """
    Create a new health record with a dummy user ID
    """
    record = HealthRecord(
        user_id=USER_ID,
        hba1c=data.hba1c,
        glucose=data.glucose,
        bmi=data.bmi,
        age=data.age
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    logger.info(f"Health record created: ID={record.id}, User={USER_ID}")
    return record


def get_health_record(db: Session, record_id: int) -> Optional[HealthRecord]:
    """Get health record by ID"""
    return db.query(HealthRecord).filter(HealthRecord.id == record_id).first()


def get_user_health_records(
    db: Session, 
    skip: int = 0, 
    limit: int = 100
) -> List[HealthRecord]:
    """Get all health records for the dummy user with pagination"""
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == USER_ID)
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


def get_latest_health_record(db: Session) -> Optional[HealthRecord]:
    """Get the most recent health record for the dummy user"""
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == USER_ID)
        .order_by(HealthRecord.created_at.desc())
        .first()
    )