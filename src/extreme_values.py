"""Deterministic extreme-value handling for the Kaggle credit dataset.

The thresholds in this module were selected from the development-only
Milestone 2D-A audit. They are dataset-specific corruption boundaries, not
universal business limits.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


EXTREME_VALUE_THRESHOLDS: Final = {
    "Num_Bank_Accounts": 11.0,
    "Num_Credit_Card": 11.0,
    "Num_of_Loan": 9.0,
    "Num_of_Delayed_Payment": 28.0,
    "Num_Credit_Inquiries": 17.0,
    "Annual_Income": 215_918.92,
}

EXTREME_INDICATOR_COLUMNS: Final = tuple(
    f"{feature}_Extreme_Invalid" for feature in EXTREME_VALUE_THRESHOLDS
)


def _validate_frame(frame: pd.DataFrame, *, operation: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise ValueError(
            f"Cannot {operation} extreme-value handling with an empty DataFrame."
        )
    if frame.columns.has_duplicates:
        raise ValueError("Input DataFrame contains duplicate column names.")

    missing = sorted(set(EXTREME_VALUE_THRESHOLDS).difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required columns for extreme-value handling: "
            f"{', '.join(missing)}"
        )

    nonnumeric = [
        column
        for column in EXTREME_VALUE_THRESHOLDS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise TypeError(
            "Extreme-value columns must be numeric after deterministic cleaning: "
            f"{', '.join(nonnumeric)}"
        )


class ExtremeValueTransformer(TransformerMixin, BaseEstimator):
    """Mask audited extreme values and add binary invalid-value indicators.

    The transformer does not learn thresholds from ``X`` and never inspects a
    target. ``fit`` stores an immutable copy of the audited rules so validation
    and final-test transformations use exactly the same thresholds.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "ExtremeValueTransformer":
        """Validate the development feature frame and freeze the audited rules."""

        _validate_frame(X, operation="fit")
        self.thresholds_ = dict(EXTREME_VALUE_THRESHOLDS)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with indicators added and extreme values set missing."""

        check_is_fitted(self, "thresholds_")
        _validate_frame(X, operation="transform")
        transformed = X.copy(deep=True)

        for feature, threshold in self.thresholds_.items():
            indicator = f"{feature}_Extreme_Invalid"
            extreme = transformed[feature].notna() & transformed[feature].gt(
                threshold
            )
            transformed[indicator] = extreme.astype(np.int8)
            transformed.loc[extreme, feature] = np.nan

        return transformed

    def get_feature_names_out(
        self, input_features: object = None
    ) -> np.ndarray:
        """Return input names followed by the six indicator names."""

        check_is_fitted(self, "thresholds_")
        if input_features is None:
            supplied = tuple(self.feature_names_in_)
        else:
            supplied = tuple(input_features)
        return np.asarray(supplied + EXTREME_INDICATOR_COLUMNS, dtype=object)
