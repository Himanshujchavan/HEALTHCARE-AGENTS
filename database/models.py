"""
Database models for the AI Health system
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database.config import Base


class User(Base):
    """
    User model for authentication
    """
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
    """
    Health record model for storing patient lab data
    """
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Lab parameters
    hba1c = Column(Float, nullable=False)
    glucose = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    
    # Additional fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    analyzed = Column(Boolean, default=False)
    analysis_result = Column(String, nullable=True)  # Store JSON analysis result
    
    # Relationship to user
    user = relationship("User", back_populates="health_records")

    def __repr__(self):
        return f"<HealthRecord(id={self.id}, user_id={self.user_id}, created_at={self.created_at})>"
