"""
CRUD operations for database models
Separates database logic from route handlers
"""
from sqlalchemy.orm import Session
from database.models import User, HealthRecord
from schemas.health_schema import HealthInput
from app.serializers import (
    normalize_assessment,
    parse_analysis_result,
    parse_extracted_parameters,
    record_assessment_item,
    serialize_assessment,
)
from typing import List, Optional, Dict, Any
import logging
import json

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

def create_health_record(
    db: Session,
    user_id: int,
    data: HealthInput,
    source: str = "manual",
    extracted_parameters: Optional[Dict[str, Any]] = None,
) -> HealthRecord:
    """
    Create a new health record for a user.

    ``source`` records where the data came from ("manual" or "report") and
    ``extracted_parameters`` keeps the raw values read from an uploaded
    report, even after the user corrects them.
    """
    record = HealthRecord(
        user_id=user_id,
        hba1c=data.hba1c,
        glucose=data.glucose,
        bmi=data.bmi,
        age=data.age,
        # Optional cardiovascular fields
        systolic_bp=getattr(data, 'systolic_bp', None),
        diastolic_bp=getattr(data, 'diastolic_bp', None),
        total_cholesterol=getattr(data, 'total_cholesterol', None),
        ldl=getattr(data, 'ldl', None),
        hdl=getattr(data, 'hdl', None),
        smoker=getattr(data, 'smoker', None),
        chest_pain_type=getattr(data, 'chest_pain_type', None),
        st_slope=getattr(data, 'st_slope', None),
        homa_ir=getattr(data, 'homa_ir', None),
        # Symptoms and manual text
        symptoms=json.dumps(data.symptoms) if data.symptoms else None,
        manual_text=data.manual_text,
        source=source,
        extracted_parameters=(
            json.dumps(extracted_parameters, default=str)
            if extracted_parameters else None
        ),
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
    analysis_result: str,
    assessment: Optional[Dict[str, Any]] = None,
    extracted_parameters: Optional[Dict[str, Any]] = None,
) -> Optional[HealthRecord]:
    """
    Update health record with analysis result and its normalized snapshot.
    """
    record = db.query(HealthRecord).filter(HealthRecord.id == record_id).first()
    if record:
        record.analyzed = True
        record.analysis_result = analysis_result
        record.assessment = serialize_assessment(
            assessment if assessment is not None else normalize_assessment(analysis_result)
        )
        if extracted_parameters is not None:
            record.extracted_parameters = json.dumps(extracted_parameters, default=str)
        db.commit()
        db.refresh(record)
        logger.info(f"Health record analyzed: ID={record_id}")
    return record


def update_health_record_values(
    db: Session,
    record: HealthRecord,
    data: HealthInput,
) -> HealthRecord:
    """Apply corrected values to a pending record."""
    record.hba1c = data.hba1c
    record.glucose = data.glucose
    record.bmi = data.bmi
    record.age = data.age
    record.systolic_bp = data.systolic_bp
    record.diastolic_bp = data.diastolic_bp
    record.total_cholesterol = data.total_cholesterol
    record.ldl = data.ldl
    record.hdl = data.hdl
    record.smoker = data.smoker
    record.chest_pain_type = data.chest_pain_type
    record.st_slope = data.st_slope
    record.homa_ir = data.homa_ir
    if data.symptoms:
        record.symptoms = json.dumps(data.symptoms)
    if data.manual_text is not None:
        record.manual_text = data.manual_text
    db.commit()
    db.refresh(record)
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
def get_history(db: Session, user_id: int, exclude_id: int | None = None, limit: int = 10) -> List[dict]:
    """Return prior health records for a user (excluding a given record).
    Each entry includes created_at, raw params, and the assessment score
    read back through the shared parser, so every record — whatever version
    wrote it — contributes the same fields.
    """
    query = db.query(HealthRecord).filter(HealthRecord.user_id == user_id)
    if exclude_id is not None:
        query = query.filter(HealthRecord.id != exclude_id)
    history = (
        query.order_by(HealthRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    history.reverse()
    result: List[dict] = []
    for rec in history:
        entry: dict = {"created_at": rec.created_at, "params": {}}
        for field in ["hba1c", "glucose", "bmi", "homa_ir", "age", "systolic_bp", "diastolic_bp", "total_cholesterol", "ldl", "hdl", "smoker", "chest_pain_type", "st_slope"]:
            value = getattr(rec, field, None)
            if value is not None:
                entry["params"][field] = value
        if rec.analysis_result:
            assessment = normalize_assessment(rec.analysis_result)
            score = assessment.get("score")
            if score is not None:
                entry["risk_score"] = score
            if assessment.get("risk_level"):
                entry["risk_level"] = assessment["risk_level"]
        result.append(entry)
    return result


def get_assessment_history(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    include_pending: bool = False,
) -> List[HealthRecord]:
    """Return the user's records, newest first, for the history endpoints."""
    query = db.query(HealthRecord).filter(HealthRecord.user_id == user_id)
    if not include_pending:
        query = query.filter(HealthRecord.analyzed.is_(True))
    return (
        query.order_by(HealthRecord.created_at.desc(), HealthRecord.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def count_assessment_history(db: Session, user_id: int, include_pending: bool = False) -> int:
    """Count the records the history endpoints paginate over."""
    query = db.query(HealthRecord).filter(HealthRecord.user_id == user_id)
    if not include_pending:
        query = query.filter(HealthRecord.analyzed.is_(True))
    return query.count()


def get_user_assessment_series(
    db: Session, user_id: int, limit: int = 100
) -> List[HealthRecord]:
    """Return analysed records oldest-first, for building time series."""
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id == user_id)
        .filter(HealthRecord.analyzed.is_(True))
        .order_by(HealthRecord.created_at.asc(), HealthRecord.id.asc())
        .limit(limit)
        .all()
    )
