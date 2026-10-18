from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database.config import Base

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to health records
    health_records = relationship("HealthRecord", back_populates="user")


class HealthRecord(Base):
    """Health record model for storing patient lab data"""
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Lab parameters (nullable for Phase 0)
    hba1c = Column(Float, nullable=True)
    glucose = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    age = Column(Integer, nullable=True)

    # Cardiovascular parameters (optional)
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    total_cholesterol = Column(Float, nullable=True)
    ldl = Column(Float, nullable=True)
    hdl = Column(Float, nullable=True)
    smoker = Column(Boolean, nullable=True)

    # Added for Phase 0: ensure re-analysis consistency
    symptoms = Column(String, nullable=True)  # Store as JSON string
    manual_text = Column(String, nullable=True)

    # Additional fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    analyzed = Column(Boolean, default=False)
    analysis_result = Column(String, nullable=True)  # Store JSON analysis result

    # Relationship to user
    user = relationship("User", back_populates="health_records")

    def __repr__(self):
        return f"<HealthRecord(id={self.id}, user_id={self.user_id}, created_at={self.created_at})>"
