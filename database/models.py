"""
Database models for the AI Health system (auth-free)
Only HealthRecord is used; User model removed for testing
"""
from sqlalchemy import Column, Integer, Float, DateTime, String, Boolean
from datetime import datetime
from database.config import Base


class HealthRecord(Base):
    """
    Health record model for storing patient lab data
    """
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, default=1)  # dummy user ID for testing
    
    # Lab parameters
    hba1c = Column(Float, nullable=False)
    glucose = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    
    # Additional fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    analyzed = Column(Boolean, default=False)
    analysis_result = Column(String, nullable=True)  # Store JSON analysis result

    def __repr__(self):
        return f"<HealthRecord(id={self.id}, user_id={self.user_id}, created_at={self.created_at})>"