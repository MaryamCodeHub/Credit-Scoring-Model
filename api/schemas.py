"""
Pydantic Schemas — Credit Scoring API.

Defines strict request/response models for input validation
and automatic OpenAPI documentation generation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ──────────────────────────────────────────────
# Enums for constrained inputs
# ──────────────────────────────────────────────

class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class EducationEnum(str, Enum):
    INTERMEDIATE = "Intermediate"
    ASSOCIATES = "Associate's Degree"
    BACHELORS = "Bachelor's Degree"
    MASTERS = "Master's Degree"
    DOCTORATE = "Doctorate"


class MaritalStatusEnum(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"


class HomeOwnershipEnum(str, Enum):
    RENTED = "Rented"
    OWNED = "Owned"


class CreditScoreEnum(str, Enum):
    LOW = "Low"
    AVERAGE = "Average"
    HIGH = "High"


class RiskLevelEnum(str, Enum):
    HIGH_RISK = "High Risk"
    MEDIUM_RISK = "Medium Risk"
    LOW_RISK = "Low Risk"


# ──────────────────────────────────────────────
# Request Schema
# ──────────────────────────────────────────────

class CreditScoreRequest(BaseModel):
    """Input schema for credit score prediction."""

    Age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant's age in years",
        examples=[30],
    )
    Gender: GenderEnum = Field(
        ...,
        description="Applicant's gender",
        examples=["Male"],
    )
    Income: float = Field(
        ...,
        gt=0,
        description="Annual income in USD",
        examples=[75000],
    )
    Education: EducationEnum = Field(
        ...,
        description="Highest education level attained",
        examples=["Bachelor's Degree"],
    )
    Marital_Status: MaritalStatusEnum = Field(
        ...,
        alias="Marital Status",
        description="Current marital status",
        examples=["Single"],
    )
    Number_of_Children: int = Field(
        ...,
        ge=0,
        le=15,
        alias="Number of Children",
        description="Number of dependent children",
        examples=[0],
    )
    Home_Ownership: HomeOwnershipEnum = Field(
        ...,
        alias="Home Ownership",
        description="Housing ownership status",
        examples=["Rented"],
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 30,
                    "Gender": "Male",
                    "Income": 75000,
                    "Education": "Bachelor's Degree",
                    "Marital Status": "Single",
                    "Number of Children": 0,
                    "Home Ownership": "Rented",
                }
            ]
        },
    }


# ──────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────

class ProbabilityBreakdown(BaseModel):
    """Probability for each credit score class."""
    Low: float = Field(..., description="Probability of Low credit score")
    Average: float = Field(..., description="Probability of Average credit score")
    High: float = Field(..., description="Probability of High credit score")


class CreditScoreResponse(BaseModel):
    """Output schema for credit score prediction."""

    credit_score: CreditScoreEnum = Field(
        ...,
        description="Predicted credit score classification",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence (probability of predicted class)",
    )
    probabilities: ProbabilityBreakdown = Field(
        ...,
        description="Probability breakdown for each credit score class",
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Risk assessment based on predicted credit score",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "credit_score": "High",
                    "confidence": 0.87,
                    "probabilities": {
                        "Low": 0.05,
                        "Average": 0.08,
                        "High": 0.87,
                    },
                    "risk_level": "Low Risk",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded and ready")
    version: str = Field(..., description="API version")
