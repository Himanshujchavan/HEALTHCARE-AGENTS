# Pydantic schemas for health data validation and API responses

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ═════════════════════════════════════════════════════════════════════════════
#  INPUT SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

#: Physiological bounds for every clinical parameter the agents accept.
#: Values outside these ranges are rejected as invalid input rather than
#: silently reaching the risk models.
PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    "hba1c": {"min": 3.0, "max": 20.0, "unit": "%", "label": "HbA1c"},
    "glucose": {"min": 40.0, "max": 600.0, "unit": "mg/dL", "label": "Fasting glucose"},
    "bmi": {"min": 10.0, "max": 70.0, "unit": "kg/m²", "label": "Body mass index"},
    "homa_ir": {"min": 0.0, "max": 20.0, "unit": "index", "label": "HOMA-IR"},
    "age": {"min": 1, "max": 120, "unit": "years", "label": "Age"},
    "systolic_bp": {"min": 50.0, "max": 260.0, "unit": "mmHg", "label": "Systolic blood pressure"},
    "diastolic_bp": {"min": 20.0, "max": 160.0, "unit": "mmHg", "label": "Diastolic blood pressure"},
    "total_cholesterol": {"min": 50.0, "max": 500.0, "unit": "mg/dL", "label": "Total cholesterol"},
    "ldl": {"min": 0.0, "max": 400.0, "unit": "mg/dL", "label": "LDL cholesterol"},
    "hdl": {"min": 10.0, "max": 200.0, "unit": "mg/dL", "label": "HDL cholesterol"},
    # Ordinal categories from the cardiovascular dataset. They are codes, not
    # measurements, so the range only bounds the code.
    "chest_pain_type": {"min": 0, "max": 3, "unit": "category", "label": "Chest pain category", "type": "integer"},
    "st_slope": {"min": 0, "max": 3, "unit": "category", "label": "ST segment slope category", "type": "integer"},
}

#: Parameters the cardiovascular module can consume.
CARDIOVASCULAR_INPUTS = [
    "age", "systolic_bp", "diastolic_bp", "total_cholesterol",
    "ldl", "hdl", "smoker", "glucose", "bmi",
    "chest_pain_type", "st_slope",
]

#: Parameters the diabetes module can consume.
DIABETES_INPUTS = ["hba1c", "glucose", "bmi", "age", "homa_ir"]


class HealthInput(BaseModel):
    """Health data submitted by the client.

    Every clinical field is optional so a partial record (a PDF extraction
    preview, for example) can be represented, but the model validator refuses
    a submission that cannot support at least one disease assessment: the
    Master Health Agent can only assess a disease whose required parameters
    are all present.
    """
    # ── Diabetes related (optional) ──────────────────────────────────────
    hba1c: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["hba1c"]["min"],
        le=PARAMETER_RANGES["hba1c"]["max"],
        description="HbA1c level in percentage (3-20)"
    )
    glucose: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["glucose"]["min"],
        le=PARAMETER_RANGES["glucose"]["max"],
        description="Blood glucose level in mg/dL (40-600)"
    )
    bmi: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["bmi"]["min"],
        le=PARAMETER_RANGES["bmi"]["max"],
        description="Body Mass Index (10-70)"
    )
    homa_ir: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["homa_ir"]["min"],
        le=PARAMETER_RANGES["homa_ir"]["max"],
        description="HOMA-IR insulin resistance index (0-20)"
    )
    age: Optional[int] = Field(
        None,
        ge=PARAMETER_RANGES["age"]["min"],
        le=PARAMETER_RANGES["age"]["max"],
        description="Patient age in years (1-120)"
    )
    # ── Cardiovascular parameters (optional) ──────────────────────────────
    systolic_bp: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["systolic_bp"]["min"],
        le=PARAMETER_RANGES["systolic_bp"]["max"],
        description="Systolic blood pressure (mmHg)"
    )
    diastolic_bp: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["diastolic_bp"]["min"],
        le=PARAMETER_RANGES["diastolic_bp"]["max"],
        description="Diastolic blood pressure (mmHg)"
    )
    total_cholesterol: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["total_cholesterol"]["min"],
        le=PARAMETER_RANGES["total_cholesterol"]["max"],
        description="Total cholesterol (mg/dL)"
    )
    ldl: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["ldl"]["min"],
        le=PARAMETER_RANGES["ldl"]["max"],
        description="LDL cholesterol (mg/dL)"
    )
    hdl: Optional[float] = Field(
        None,
        ge=PARAMETER_RANGES["hdl"]["min"],
        le=PARAMETER_RANGES["hdl"]["max"],
        description="HDL cholesterol (mg/dL)"
    )
    smoker: Optional[bool] = Field(
        None,
        description=(
            "Smoking status. Omit it when the status is unknown: an absent "
            "value means 'not supplied', not 'does not smoke', and the rule "
            "scorer reports the two differently."
        )
    )
    chest_pain_type: Optional[int] = Field(
        None,
        ge=PARAMETER_RANGES["chest_pain_type"]["min"],
        le=PARAMETER_RANGES["chest_pain_type"]["max"],
        description=(
            "Chest pain category code (0-3) from the cardiovascular dataset. "
            "Optional: the cardiovascular model uses it only together with "
            "st_slope, and otherwise falls back to its base feature set."
        )
    )
    st_slope: Optional[int] = Field(
        None,
        ge=PARAMETER_RANGES["st_slope"]["min"],
        le=PARAMETER_RANGES["st_slope"]["max"],
        description=(
            "ST segment slope category code (0-3) from the cardiovascular "
            "dataset. Optional: the cardiovascular model uses it only "
            "together with chest_pain_type."
        )
    )
    # ── Context ───────────────────────────────────────────────────────────
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

    @field_validator('hba1c', 'glucose', 'bmi', 'homa_ir')
    @classmethod
    def validate_positive_numbers(cls, v):
        """Ensure values are positive when provided"""
        if v is None:
            return v
        if v <= 0:
            raise ValueError("Value must be positive")
        return round(v, 2)

    @field_validator('systolic_bp', 'diastolic_bp', 'total_cholesterol', 'ldl', 'hdl')
    @classmethod
    def round_clinical_values(cls, v):
        """Clinical values are stored with two decimals."""
        return None if v is None else round(v, 2)

    @field_validator('symptoms')
    @classmethod
    def clean_symptoms(cls, v):
        """Drop blank entries and keep the list free of duplicates."""
        seen, cleaned = set(), []
        for symptom in v or []:
            text = str(symptom).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def validate_clinical_consistency(self):
        """Reject values that are individually valid but mutually impossible."""
        if (
            self.systolic_bp is not None
            and self.diastolic_bp is not None
            and self.diastolic_bp >= self.systolic_bp
        ):
            raise ValueError(
                "diastolic_bp must be lower than systolic_bp "
                f"(got {self.diastolic_bp} >= {self.systolic_bp})"
            )

        if (
            self.total_cholesterol is not None
            and self.ldl is not None
            and self.ldl > self.total_cholesterol
        ):
            raise ValueError(
                "ldl cannot be higher than total_cholesterol "
                f"(got {self.ldl} > {self.total_cholesterol})"
            )

        if (
            self.total_cholesterol is not None
            and self.hdl is not None
            and self.ldl is not None
            and self.hdl + self.ldl > self.total_cholesterol + 30
        ):
            raise ValueError(
                "hdl + ldl cannot exceed total_cholesterol by more than 30 mg/dL "
                f"(got {self.hdl} + {self.ldl} > {self.total_cholesterol})"
            )
        return self

    @model_validator(mode="after")
    def require_supported_disease(self):
        """At least one disease must be fully described by the submission.

        The requirement is read from the disease plugins themselves, so the
        API can never drift from what the agents actually require.
        """
        from Agents.plugins import registry

        data = self.model_dump()
        complete, missing = [], {}
        for name in registry.list_all():
            plugin = registry.get(name)
            if plugin is None:
                continue
            gaps = registry.missing_fields(data, plugin)
            if gaps:
                missing[name] = gaps
            else:
                complete.append(name)

        if not complete:
            detail = "; ".join(
                f"{disease} is missing {', '.join(fields)}"
                for disease, fields in missing.items()
            )
            raise ValueError(
                "Provide every required parameter for at least one condition "
                f"({detail or 'no condition could be assessed'})"
            )
        return self

    def missing_fields_by_disease(self) -> Dict[str, List[str]]:
        """Required-but-absent parameters, per disease."""
        from Agents.plugins import registry

        data = self.model_dump()
        result: Dict[str, List[str]] = {}
        for name in registry.list_all():
            plugin = registry.get(name)
            if plugin is None:
                continue
            gaps = registry.missing_fields(data, plugin)
            if gaps:
                result[name] = gaps
        return result


class ExtractionConfirmRequest(HealthInput):
    """Corrected values submitted after a report extraction preview."""
    extraction_id: int = Field(
        ...,
        description="Identifier of the pending extraction returned by the preview"
    )
    confirmed_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The values the user kept or corrected, keyed by parameter"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  NORMALIZED RESPONSE SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class AgentStatusModel(BaseModel):
    """Status of a single agent inside the workflow."""
    status: str
    reason: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class DiseaseAssessment(BaseModel):
    """Normalized per-disease assessment."""
    status: str
    risk_level: Optional[str] = None
    score: Optional[float] = None
    factors: List[str] = Field(default_factory=list)
    agent_status: Dict[str, AgentStatusModel] = Field(default_factory=dict)
    analysis: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    symptoms: Optional[Dict[str, Any]] = None
    trend: Optional[Dict[str, Any]] = None
    final_assessment: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class AlertSummary(BaseModel):
    """A single alert emitted by the alert agent."""
    severity: str
    alert: bool
    explanation: Optional[str] = None
    recommended_next_steps: Optional[str] = None
    triggers: List[Any] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class DifferentialSummary(BaseModel):
    """Differential diagnosis outcome across the assessed diseases."""
    status: str
    results: Any = None

    model_config = ConfigDict(extra="allow")


class TrendSummary(BaseModel):
    """Trend outcome. Carries the raw trend payload when one was computed."""
    status: Optional[str] = None
    trend_score: Optional[float] = None
    direction: Optional[str] = None
    visits_used: Optional[int] = None
    changes: Dict[str, Any] = Field(default_factory=dict)
    results: Any = None
    reason: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class UnifiedAssessmentResponse(BaseModel):
    """The single response shape every assessment endpoint returns."""
    assessment_id: Optional[int] = None
    status: str
    assessments: Dict[str, DiseaseAssessment] = Field(default_factory=dict)
    differential: DifferentialSummary = Field(default_factory=lambda: DifferentialSummary(status="unavailable"))
    trend: TrendSummary = Field(default_factory=lambda: TrendSummary(status="unavailable"))
    alerts: List[AlertSummary] = Field(default_factory=list)
    explanation: Optional[Any] = None
    agent_status: Dict[str, AgentStatusModel] = Field(default_factory=dict)
    created_at: Optional[str] = None
    model_disclaimer: str = (
        "These are model predictions and risk signals, not confirmed medical diagnoses."
    )

    model_config = ConfigDict(extra="allow")


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
    chest_pain_type: Optional[int] = None
    st_slope: Optional[int] = None
    source: str = "manual"
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
    # Workflow-contract names, so the API mirrors the agent output keys
    trend_result: Optional[Dict[str, Any]] = None
    differential_result: Optional[Dict[str, Any]] = None
    disease_results: Optional[Dict[str, Any]] = None
    agent_status: Optional[Dict[str, Any]] = None
    assessment: Optional[Dict[str, Any]] = None
    unified: Optional[UnifiedAssessmentResponse] = None

    class Config:
        from_attributes = True


class ExtractedParameter(BaseModel):
    """One parameter read from a report, with its clinical status."""
    value: Optional[float] = None
    status: str = "Unknown"
    unit: str = ""
    note: Optional[str] = None
    corrected: bool = False

    model_config = ConfigDict(extra="allow")


class ExtractionPreviewResponse(BaseModel):
    """Preview of the values extracted from an uploaded report.

    The record is created in a ``pending`` state: no assessment is run until
    the client confirms the values.
    """
    status: str
    message: str
    extraction_id: int
    filename: str
    source: str = "report"
    extracted_parameters: Dict[str, ExtractedParameter] = Field(default_factory=dict)
    missing_fields: Dict[str, List[str]] = Field(default_factory=dict)
    available_diseases: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None

    model_config = ConfigDict(extra="allow")


class AssessmentHistoryItem(BaseModel):
    """One assessment in the user's history."""
    record_id: int
    created_at: datetime
    source: str = "manual"
    status: str
    disease: Optional[str] = None
    risk_level: Optional[str] = None
    score: Optional[float] = None
    diseases: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agent_status: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class AssessmentHistoryResponse(BaseModel):
    """Paginated assessment history."""
    status: str
    count: int
    total: int
    skip: int
    limit: int
    items: List[AssessmentHistoryItem] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class AssessmentSeriesPoint(BaseModel):
    """A single point of a parameter or score time series."""
    record_id: int
    created_at: Optional[datetime] = None
    value: Optional[float] = None
    risk_level: Optional[str] = None
    disease: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class AssessmentHistorySummary(BaseModel):
    """Time series of the user's assessments, for charting."""
    status: str
    count: int
    parameters: Dict[str, List[AssessmentSeriesPoint]] = Field(default_factory=dict)
    scores: List[AssessmentSeriesPoint] = Field(default_factory=list)
    risk_levels: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class InputSchemaResponse(BaseModel):
    """Describes the parameters the API accepts, so the client can render
    exactly the inputs the agents require."""
    status: str
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    diseases: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    symptoms: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ErrorDetail(BaseModel):
    """A single validation problem."""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    """Normalized error envelope returned for every rejected request."""
    status: str = "error"
    error: str
    message: Optional[str] = None
    details: List[ErrorDetail] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

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
