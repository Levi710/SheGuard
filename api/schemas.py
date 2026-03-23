"""
MamaGuard — API Schemas
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class RiskLevelEnum(str, Enum):
    LOW    = "Low risk"
    MEDIUM = "Medium risk"
    HIGH   = "High risk"


class PrenatalVisit(BaseModel):
    """One prenatal checkup reading."""
    age:           float = Field(...,  ge=10,  le=60,  description="Patient age in years")
    systolic_bp:   float = Field(...,  ge=70,  le=200, description="Systolic blood pressure mmHg")
    diastolic_bp:  float = Field(...,  ge=40,  le=130, description="Diastolic blood pressure mmHg")
    blood_sugar:   Optional[float] = Field(None, ge=3.0, le=20.0, description="Blood glucose mmol/L")
    body_temp:     Optional[float] = Field(None, ge=35.0, le=42.0, description="Temperature °C")
    heart_rate:    Optional[float] = Field(None, ge=40,  le=160,  description="Heart rate bpm")
    visit_date:    Optional[str]   = Field(None, description="ISO date of visit e.g. 2024-03-15")


class PredictionRequest(BaseModel):
    """Request body for POST /predict."""
    patient_id:      str            = Field(..., description="Clinic's own patient ID")
    visits:          List[PrenatalVisit] = Field(
        ..., min_length=1, max_length=10,
        description="Prenatal visit readings in chronological order (oldest first)"
    )
    staff_available: Optional[int]  = Field(None, ge=0, description="Doctors on duty right now")
    blood_units:     Optional[int]  = Field(None, ge=0, description="Units of O-negative blood in stock")
    clinic_name:     Optional[str]  = Field(None, description="Clinic name for alert routing")

    @field_validator("visits")
    @classmethod
    def at_least_one_visit(cls, v):
        if len(v) < 1:
            raise ValueError("At least 1 prenatal visit is required")
        return v


class AlertTier(str, Enum):
    GREEN  = "GREEN"   # < 65% high risk confidence
    AMBER  = "AMBER"   # 65–85% — watch closely
    RED    = "RED"     # > 85% + 2+ consecutive high-risk visits


class PredictionResponse(BaseModel):
    """Response from POST /predict."""
    patient_id:       str
    risk_level:       RiskLevelEnum
    alert_tier:       AlertTier
    confidence:       float
    data_quality:     float
    top_reasons:      List[str]
    feature_importance: dict
    visit_importance: List[float]
    action_required:  str
    transfer_order:   Optional[str] = None
    suppressed:       bool = False
    probabilities:    dict