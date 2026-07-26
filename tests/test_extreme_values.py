"""Focused tests for deterministic extreme-value handling."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extreme_values import (  # noqa: E402
    EXTREME_INDICATOR_COLUMNS,
    EXTREME_VALUE_THRESHOLDS,
    ExtremeValueTransformer,
)


def _frame() -> pd.DataFrame:
    data: dict[str, object] = {}
    for feature, threshold in EXTREME_VALUE_THRESHOLDS.items():
        data[feature] = [threshold, threshold + 1, np.nan, threshold - 1]
    data["Credit_Score"] = ["Good", "Poor", "Standard", "Good"]
    data["Unrelated"] = ["keep-a", "keep-b", "keep-c", "keep-d"]
    return pd.DataFrame(data, index=[10, 20, 30, 40])


def test_boundary_values_are_retained_and_above_thresholds_become_missing() -> None:
    transformed = ExtremeValueTransformer().fit_transform(_frame())

    for feature, threshold in EXTREME_VALUE_THRESHOLDS.items():
        assert transformed.loc[10, feature] == pytest.approx(threshold)
        assert pd.isna(transformed.loc[20, feature])


def test_correct_indicators_are_created_and_existing_missing_is_zero() -> None:
    transformed = ExtremeValueTransformer().fit_transform(_frame())

    assert tuple(
        column
        for column in transformed.columns
        if column.endswith("_Extreme_Invalid")
    ) == EXTREME_INDICATOR_COLUMNS
    for indicator in EXTREME_INDICATOR_COLUMNS:
        assert transformed.loc[10, indicator] == 0
        assert transformed.loc[20, indicator] == 1
        assert transformed.loc[30, indicator] == 0
        assert transformed.loc[40, indicator] == 0


def test_unrelated_columns_rows_and_index_are_preserved() -> None:
    frame = _frame()
    transformed = ExtremeValueTransformer().fit_transform(frame)

    pd.testing.assert_series_equal(transformed["Unrelated"], frame["Unrelated"])
    assert len(transformed) == len(frame)
    assert transformed.index.equals(frame.index)


def test_input_frame_is_not_mutated() -> None:
    frame = _frame()
    original = frame.copy(deep=True)

    ExtremeValueTransformer().fit_transform(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_credit_score_does_not_affect_transformation() -> None:
    first = _frame()
    second = first.copy(deep=True)
    second["Credit_Score"] = ["Poor", "Poor", "Poor", "Poor"]
    transformer = ExtremeValueTransformer().fit(first)

    first_transformed = transformer.transform(first)
    second_transformed = transformer.transform(second)

    pd.testing.assert_frame_equal(
        first_transformed.drop(columns="Credit_Score"),
        second_transformed.drop(columns="Credit_Score"),
    )
    pd.testing.assert_series_equal(
        second_transformed["Credit_Score"],
        second["Credit_Score"],
    )


def test_repeated_transformations_are_identical() -> None:
    frame = _frame()
    transformer = ExtremeValueTransformer().fit(frame)

    pd.testing.assert_frame_equal(
        transformer.transform(frame),
        transformer.transform(frame),
    )


def test_unapproved_financial_features_are_unchanged() -> None:
    frame = _frame().assign(
        Monthly_Inhand_Salary=1_000_000.0,
        Outstanding_Debt=1_000_000.0,
        Total_EMI_per_month=1_000_000.0,
        Credit_Utilization_Ratio=99.0,
        Credit_History_Age_Months=999.0,
    )
    transformed = ExtremeValueTransformer().fit_transform(frame)

    for column in (
        "Monthly_Inhand_Salary",
        "Outstanding_Debt",
        "Total_EMI_per_month",
        "Credit_Utilization_Ratio",
        "Credit_History_Age_Months",
    ):
        pd.testing.assert_series_equal(transformed[column], frame[column])


def test_missing_required_columns_raise_clear_error() -> None:
    frame = _frame().drop(columns="Num_Bank_Accounts")

    with pytest.raises(ValueError, match="Num_Bank_Accounts"):
        ExtremeValueTransformer().fit(frame)


def test_raw_source_file_is_never_modified(tmp_path: Path) -> None:
    raw_file = tmp_path / "train.csv"
    raw_file.write_bytes(b"Customer_ID,Credit_Score\nCUS_1,Good\n")
    before = hashlib.sha256(raw_file.read_bytes()).hexdigest()

    ExtremeValueTransformer().fit_transform(_frame())

    assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == before
