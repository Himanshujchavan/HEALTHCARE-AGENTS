"""
Pydantic schemas for health data validation
Ensures input data meets medical and technical requirements
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class HealthInput(BaseModel):
    """
    Schema for validating health data input
    Prevents invalid values and malicious input
    """
    hba1c: float = Field(
        ..., 
        ge=3.0, 
        le=20.0,
        description="HbA1c level in percentage (3-20)"
    )
    glucose: float = Field(
        ..., 
        ge=50.0, 
        le=400.0,
        description="Blood glucose level in mg/dL (50-400)"
    )
    bmi: float = Field(
        ..., 
        ge=10.0, 
        le=60.0,
        description="Body Mass Index (10-60)"
    )
    age: int = Field(
        ..., 
        ge=1, 
        le=120,
        description="Patient age in years (1-120)"
    )

    @field_validator('hba1c', 'glucose', 'bmi')
    @classmethod
    def validate_positive_numbers(cls, v):
        """Ensure values are positive"""
        if v <= 0:
            raise ValueError("Value must be positive")
        return round(v, 2)  # Round to 2 decimal places

    class Config:
        json_schema_extra = {
            "example": {
                "hba1c": 6.8,
                "glucose": 148.0,
                "bmi": 29.0,
                "age": 45
            }
        }


class HealthRecordResponse(BaseModel):
    """
    Response schema for health record
    """
    id: int
    user_id: int
    hba1c: float
    glucose: float
    bmi: float
    age: int
    created_at: datetime
    analyzed: bool
    analysis_result: Optional[str] = None

    class Config:
        from_attributes = True


class HealthDataSubmitResponse(BaseModel):
    """
    Response after submitting health data
    """
    status: str
    message: str
    record_id: int
    timestamp: datetime


class UserCreate(BaseModel):
    """
    Schema for user registration
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """
    Schema for user login
    """
    username: str
    password: str


class UserResponse(BaseModel):
    """
    Response schema for user data
    """
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """
    JWT token response
    """
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """
    Data contained in JWT token
    """
    user_id: Optional[int] = None
    username: Optional[str] = None
