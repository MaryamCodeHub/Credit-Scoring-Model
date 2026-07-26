"""Leakage-safe feature selection and preprocessing for credit-score modeling.

This module is intentionally separate from the legacy production preprocessing
code. It defines the Milestone 2C feature contract and provides an in-memory
scikit-learn transformer that must be fitted on development data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from src.extreme_values import (
    EXTREME_INDICATOR_COLUMNS,
    ExtremeValueTransformer,
)


TARGET_COLUMN: Final = "Credit_Score"
GROUP_COLUMN: Final = "Customer_ID"

FORBIDDEN_COLUMNS: Final = (
    "ID",
    GROUP_COLUMN,
    "Name",
    "SSN",
    TARGET_COLUMN,
)

QUARANTINED_COLUMNS: Final = (
    "Credit_Mix",
    "Month",
    "Interest_Rate",
    "Delay_from_due_date",
    "Payment_of_Min_Amount",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
    "Type_of_Loan",
)

INPUT_NUMERICAL_FEATURES: Final = (
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age_Months",
    "Total_EMI_per_month",
)

NUMERICAL_FEATURES: Final = (
    INPUT_NUMERICAL_FEATURES + EXTREME_INDICATOR_COLUMNS
)
CATEGORICAL_FEATURES: Final = ("Occupation",)
MODEL_INPUT_FEATURES: Final = INPUT_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
MODEL_FEATURES: Final = MODEL_INPUT_FEATURES


@dataclass(frozen=True)
class ModelInputs:
    """Aligned modeling inputs selected from one cleaned data partition."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


def separate_features_target_groups(frame: pd.DataFrame) -> ModelInputs:
    """Return approved features, target, and customer groups with aligned indices.

    The input may contain raw identifiers and quarantined columns because this
    function selects only the explicitly approved model features. No input data
    is mutated.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Cannot separate model inputs from an empty DataFrame.")
    if frame.columns.has_duplicates:
        raise ValueError("Input DataFrame contains duplicate column names.")

    required = MODEL_INPUT_FEATURES + (TARGET_COLUMN, GROUP_COLUMN)
    missing = _missing_columns(frame, required)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return ModelInputs(
        X=frame.loc[:, MODEL_INPUT_FEATURES].copy(deep=True),
        y=frame.loc[:, TARGET_COLUMN].copy(deep=True),
        groups=frame.loc[:, GROUP_COLUMN].copy(deep=True),
    )


def _validated_feature_copy(frame: pd.DataFrame, *, operation: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected model features as a pandas DataFrame.")
    if frame.empty:
        raise ValueError(f"Cannot {operation} preprocessing with an empty DataFrame.")
    if frame.columns.has_duplicates:
        raise ValueError("Model feature DataFrame contains duplicate column names.")

    present_forbidden = sorted(set(frame.columns).intersection(FORBIDDEN_COLUMNS))
    if present_forbidden:
        raise ValueError(
            "Forbidden identifier, PII, grouping, or target columns entered the "
            f"model feature matrix: {', '.join(present_forbidden)}"
        )

    present_quarantined = sorted(set(frame.columns).intersection(QUARANTINED_COLUMNS))
    if present_quarantined:
        raise ValueError(
            "Quarantined columns entered the model feature matrix: "
            f"{', '.join(present_quarantined)}"
        )

    missing = _missing_columns(frame, MODEL_INPUT_FEATURES)
    if missing:
        raise ValueError(f"Missing required model feature columns: {', '.join(missing)}")

    unexpected = sorted(set(frame.columns).difference(MODEL_INPUT_FEATURES))
    if unexpected:
        raise ValueError(f"Unexpected model feature columns: {', '.join(unexpected)}")

    result = frame.loc[:, MODEL_INPUT_FEATURES].copy(deep=True)
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def _make_column_transformer() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, list(NUMERICAL_FEATURES)),
            ("occupation", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _make_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("extreme_values", ExtremeValueTransformer()),
            ("columns", _make_column_transformer()),
        ]
    )


class CreditModelPreprocessor(TransformerMixin, BaseEstimator):
    """Fit development-only imputers and encoding for the approved features."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "CreditModelPreprocessor":
        """Fit learned preprocessing state; ``y`` is accepted but never used."""

        validated = _validated_feature_copy(X, operation="fit")
        self.pipeline_ = _make_preprocessing_pipeline()
        self.pipeline_.fit(validated)
        self.extreme_value_transformer_ = self.pipeline_.named_steps[
            "extreme_values"
        ]
        self.column_transformer_ = self.pipeline_.named_steps["columns"]
        self.feature_names_in_ = np.asarray(MODEL_INPUT_FEATURES, dtype=object)
        self.n_features_in_ = len(MODEL_INPUT_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using already-fitted development statistics."""

        check_is_fitted(self, "pipeline_")
        validated = _validated_feature_copy(X, operation="transform")
        return np.asarray(self.pipeline_.transform(validated))

    def get_feature_names_out(
        self, input_features: object = None
    ) -> np.ndarray:
        """Return stable output names after the preprocessor has been fitted."""

        check_is_fitted(self, "pipeline_")
        if input_features is not None:
            supplied = tuple(input_features)
            if supplied != MODEL_INPUT_FEATURES:
                raise ValueError("input_features must match the approved feature contract.")
        return self.column_transformer_.get_feature_names_out()


def build_model_preprocessor() -> CreditModelPreprocessor:
    """Create a new, unfitted Milestone 2C preprocessor."""

    return CreditModelPreprocessor()
