"""Strict public request and response schemas for model v2."""

from __future__ import annotations

import unicodedata
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


FiniteNonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FinitePercentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
FiniteSignedFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInteger = Annotated[int, Field(strict=True, ge=0)]


class CreditScoreV2Request(BaseModel):
    """Validated monthly financial profile accepted by the v2 endpoint."""

    model_config = ConfigDict(extra="forbid")

    age: Annotated[int, Field(strict=True, ge=18, le=120)]
    annual_income: FiniteNonNegativeFloat
    monthly_inhand_salary: FiniteNonNegativeFloat | None = None
    num_bank_accounts: NonNegativeInteger
    num_credit_cards: NonNegativeInteger
    num_loans: NonNegativeInteger
    num_delayed_payments: NonNegativeInteger
    changed_credit_limit: FiniteSignedFloat | None = None
    num_credit_inquiries: NonNegativeInteger | None = None
    outstanding_debt: FiniteNonNegativeFloat
    credit_utilization_ratio: FinitePercentage
    credit_history_age_months: NonNegativeInteger
    total_emi_per_month: FiniteNonNegativeFloat
    occupation: Annotated[str, Field(min_length=1, max_length=100)] | None = None

    MODEL_FIELD_MAP: ClassVar[dict[str, str]] = {
        "age": "Age",
        "annual_income": "Annual_Income",
        "monthly_inhand_salary": "Monthly_Inhand_Salary",
        "num_bank_accounts": "Num_Bank_Accounts",
        "num_credit_cards": "Num_Credit_Card",
        "num_loans": "Num_of_Loan",
        "num_delayed_payments": "Num_of_Delayed_Payment",
        "changed_credit_limit": "Changed_Credit_Limit",
        "num_credit_inquiries": "Num_Credit_Inquiries",
        "outstanding_debt": "Outstanding_Debt",
        "credit_utilization_ratio": "Credit_Utilization_Ratio",
        "credit_history_age_months": "Credit_History_Age_Months",
        "total_emi_per_month": "Total_EMI_per_month",
        "occupation": "Occupation",
    }

    @field_validator("occupation", mode="before")
    @classmethod
    def validate_occupation(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("occupation must not be blank")
        if any(unicodedata.category(character) == "Cc" for character in trimmed):
            raise ValueError("occupation must not contain control characters")
        return trimmed

    def to_model_features(self) -> dict[str, object]:
        """Map public fields to the frozen pipeline's exact feature names."""
        public_values = self.model_dump()
        return {
            model_name: public_values[public_name]
            for public_name, model_name in self.MODEL_FIELD_MAP.items()
        }


class CreditScoreV2Probabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poor: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    standard: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    good: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class CreditScoreV2Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credit_score: Annotated[str, Field(pattern="^(Poor|Standard|Good)$")]
    probabilities: CreditScoreV2Probabilities
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    risk_level: Annotated[
        str, Field(pattern="^(High Risk|Medium Risk|Low Risk)$")
    ]
    model_version: Annotated[str, Field(pattern=r"^2\.0\.0$")]
