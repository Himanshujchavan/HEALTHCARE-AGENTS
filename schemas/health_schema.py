# Pydantic schemas for health data validation and API responses

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class HealthInput(BaseModel):
    """Schema for validating health data input.
    Optional fields allow partial submissions; validation ensures positive numeric values.
    """
    # Diabetes related (optional)
    hba1c: Optional[float] = Field(
        None,
        ge=3.0,
        le=20.0,
        description="HbA1c level in percentage (3-20)"
    )
    glucose: Optional[float] = Field(
        None,
        ge=50.0,
        le=400.0,
        description="Blood glucose level in mg/dL (50-400)"
    )
    bmi: Optional[float] = Field(
        None,
        ge=10.0,
        le=60.0,
        description="Body Mass Index (10-60)"
    )
    age: Optional[int] = Field(
        None,
        ge=1,
        le=120,
        description="Patient age in years (1-120)"
    )
    # Cardiovascular parameters (optional)
    systolic_bp: Optional[float] = Field(
        None,
        ge=80.0,
        le=200.0,
        description="Systolic blood pressure (mmHg)"
    )
    diastolic_bp: Optional[float] = Field(
        None,
        ge=40.0,
        le=130.0,
        description="Diastolic blood pressure (mmHg)"
    )
    total_cholesterol: Optional[float] = Field(
        None,
        ge=100.0,
        le=400.0,
        description="Total cholesterol (mg/dL)"
    )
    ldl: Optional[float] = Field(
        None,
        ge=0.0,
        le=300.0,
        description="LDL cholesterol (mg/dL)"
    )
    hdl: Optional[float] = Field(
        None,
        ge=20.0,
        le=150.0,
        description="HDL cholesterol (mg/dL)"
    )
    smoker: Optional[bool] = Field(
        None,
        description="Smoking status (True if smoker)"
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Optional list of patient symptoms for symptom analysis"
    )
    manual_text: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text clinical context or patient notes"
    )
    use_llm: Optional[bool] = Field(
        default=False,
        description="Use LLM enhancement for nuanced summaries"
    )

    @field_validator('hba1c', 'glucose', 'bmi')
    @classmethod
    def validate_positive_numbers(cls, v):
        """Ensure values are positive when provided"""
        if v is None:
            return v
        if v <= 0:
            raise ValueError("Value must be positive")
        return round(v, 2)

    @model_validator(mode="after")
    def require_supported_disease(self):
        diabetes_complete = all(
            getattr(self, field) is not None
            for field in ("hba1c", "glucose", "bmi", "age")
        )
        cardiovascular_complete = all(
            getattr(self, field) is not None
            for field in ("age", "systolic_bp", "total_cholesterol", "smoker")
        )
        if not (diabetes_complete or cardiovascular_complete):
            raise ValueError(
                "Provide all required diabetes or cardiovascular parameters"
            )
        return self


class HealthRecordResponse(BaseModel):
    """Response schema for health record (raw DB row)."""
    id: int
    user_id: int
    hba1c: Optional[float] = None
    glucose: Optional[float] = None
    bmi: Optional[float] = None
    age: Optional[int] = None
    # Cardiovascular fields
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    total_cholesterol: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    smoker: Optional[bool] = None
    created_at: datetime
    analyzed: bool
    analysis_result: Optional[str] = None

    class Config:
        from_attributes = True


class HealthDataSubmitResponse(BaseModel):
    """Response after submitting health data through the full master workflow."""
    status: str
    message: str
    record_id: int
    timestamp: datetime
    workflow_status: Optional[str] = None
    risk_level: Optional[str] = None
    alert: Optional[bool] = None
    report: Optional[str] = None
    final_assessment: Optional[Dict[str, Any]] = None
    # New comprehensive fields
    disease: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    trend: Optional[Dict[str, Any]] = None
    differential: Optional[Dict[str, Any]] = None
    disease_results: Optional[Dict[str, Any]] = None
    unified: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class UnifiedAssessmentResponse(BaseModel):
    assessment_id: Optional[int] = None
    status: str
    assessments: Dict[str, Any]
    differential: Dict[str, Any]
    trend: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    explanation: Any
    created_at: Optional[datetime] = None
    model_disclaimer: str


class UserCreate(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class UserResponse(BaseModel):
    """Response schema for user data"""
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data contained in JWT token"""
    user_id: Optional[int] = None
    username: Optional[str] = None
