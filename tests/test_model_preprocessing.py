"""Focused tests for leakage-safe Milestone 2C preprocessing."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FORBIDDEN_COLUMNS,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    QUARANTINED_COLUMNS,
    build_model_preprocessor,
    separate_features_target_groups,
)


def _cleaned_frame() -> pd.DataFrame:
    rows = 4
    data: dict[str, object] = {
        column: np.arange(1, rows + 1, dtype=float)
        for column in NUMERICAL_FEATURES
    }
    data["Age"] = [20.0, 30.0, np.nan, 40.0]
    data["Occupation"] = ["Engineer", np.nan, "Teacher", "Engineer"]
    data["Credit_Score"] = ["Good", "Standard", "Poor", "Good"]
    data["Customer_ID"] = ["CUS_1", "CUS_2", "CUS_3", "CUS_4"]
    data["ID"] = ["ROW_1", "ROW_2", "ROW_3", "ROW_4"]
    data["Name"] = ["Person A", "Person B", "Person C", "Person D"]
    data["SSN"] = ["masked-1", "masked-2", "masked-3", "masked-4"]
    data["Month"] = ["January", "January", "January", "January"]
    data["Credit_Mix"] = ["Good", "Standard", "Bad", "Good"]
    data["Type_of_Loan"] = [pd.NA, "Auto Loan", "Mortgage Loan", pd.NA]
    return pd.DataFrame(data, index=[10, 20, 30, 40])


def _fitted():
    inputs = separate_features_target_groups(_cleaned_frame())
    return build_model_preprocessor().fit(inputs.X), inputs


def _state(preprocessor) -> tuple[np.ndarray, tuple[tuple[object, ...], ...]]:
    numeric = preprocessor.column_transformer_.named_transformers_[
        "numerical"
    ].named_steps["imputer"].statistics_.copy()
    categories = tuple(
        tuple(values)
        for values in preprocessor.column_transformer_.named_transformers_[
            "occupation"
        ].named_steps["encoder"].categories_
    )
    return numeric, categories


def test_feature_contract_excludes_forbidden_and_quarantined_columns() -> None:
    inputs = separate_features_target_groups(_cleaned_frame())

    assert tuple(inputs.X.columns) == MODEL_FEATURES
    assert not set(inputs.X).intersection(FORBIDDEN_COLUMNS)
    assert not set(inputs.X).intersection(QUARANTINED_COLUMNS)


def test_x_y_and_groups_keep_the_original_alignment() -> None:
    frame = _cleaned_frame()
    inputs = separate_features_target_groups(frame)

    assert inputs.X.index.equals(frame.index)
    assert inputs.y.index.equals(frame.index)
    assert inputs.groups.index.equals(frame.index)
    assert inputs.y.equals(frame["Credit_Score"])
    assert inputs.groups.equals(frame["Customer_ID"])


def test_numerical_missing_values_are_median_imputed() -> None:
    preprocessor, inputs = _fitted()
    transformed = preprocessor.transform(inputs.X)
    age_index = list(preprocessor.get_feature_names_out()).index("Age")

    assert transformed[2, age_index] == pytest.approx(30.0)
    assert not np.isnan(transformed.astype(float)).any()


def test_numerical_imputation_is_learned_only_from_development() -> None:
    development = separate_features_target_groups(_cleaned_frame()).X
    validation = development.iloc[[0]].copy()
    validation.loc[:, "Age"] = np.nan
    preprocessor = build_model_preprocessor().fit(development)
    state_before = _state(preprocessor)

    transformed = preprocessor.transform(validation)
    state_after = _state(preprocessor)
    age_index = list(preprocessor.get_feature_names_out()).index("Age")

    assert transformed[0, age_index] == pytest.approx(30.0)
    np.testing.assert_array_equal(state_after[0], state_before[0])
    assert state_after[1] == state_before[1]


def test_missing_occupation_is_encoded_as_unknown() -> None:
    preprocessor, inputs = _fitted()
    transformed = preprocessor.transform(inputs.X)
    names = list(preprocessor.get_feature_names_out())
    unknown_index = names.index("Occupation_Unknown")

    assert transformed[1, unknown_index] == pytest.approx(1.0)


def test_unseen_validation_occupation_is_ignored() -> None:
    preprocessor, inputs = _fitted()
    validation = inputs.X.iloc[[0]].copy()
    validation.loc[:, "Occupation"] = "Astronaut"

    transformed = preprocessor.transform(validation)
    names = preprocessor.get_feature_names_out()
    occupation_indices = [
        index for index, name in enumerate(names) if name.startswith("Occupation_")
    ]

    assert transformed.shape[0] == 1
    assert transformed[0, occupation_indices].sum() == pytest.approx(0.0)


def test_validation_transform_does_not_refit() -> None:
    preprocessor, inputs = _fitted()
    validation = inputs.X.iloc[[0]].copy()
    validation.loc[:, "Age"] = 1_000_000.0
    state_before = _state(preprocessor)

    preprocessor.transform(validation)

    state_after = _state(preprocessor)
    np.testing.assert_array_equal(state_after[0], state_before[0])
    assert state_after[1] == state_before[1]


def test_final_test_transform_does_not_refit() -> None:
    preprocessor, inputs = _fitted()
    final_test = inputs.X.iloc[[3]].copy()
    final_test.loc[:, "Occupation"] = "Unseen Final Occupation"
    state_before = _state(preprocessor)

    preprocessor.transform(final_test)

    state_after = _state(preprocessor)
    np.testing.assert_array_equal(state_after[0], state_before[0])
    assert state_after[1] == state_before[1]


def test_feature_names_are_stable_and_available() -> None:
    first, inputs = _fitted()
    second = build_model_preprocessor().fit(inputs.X)

    assert tuple(first.get_feature_names_out()) == tuple(second.get_feature_names_out())
    assert tuple(first.get_feature_names_out()[: len(NUMERICAL_FEATURES)]) == (
        NUMERICAL_FEATURES
    )
    assert set(CATEGORICAL_FEATURES).isdisjoint(first.get_feature_names_out())


def test_repeated_fits_are_deterministic() -> None:
    inputs = separate_features_target_groups(_cleaned_frame())
    first = build_model_preprocessor().fit(inputs.X)
    second = build_model_preprocessor().fit(inputs.X)

    np.testing.assert_array_equal(first.transform(inputs.X), second.transform(inputs.X))
    np.testing.assert_array_equal(_state(first)[0], _state(second)[0])
    assert _state(first)[1] == _state(second)[1]


def test_inputs_are_not_mutated() -> None:
    frame = _cleaned_frame()
    original_frame = frame.copy(deep=True)
    inputs = separate_features_target_groups(frame)
    original_x = inputs.X.copy(deep=True)
    preprocessor = build_model_preprocessor().fit(inputs.X)

    preprocessor.transform(inputs.X)

    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_frame_equal(inputs.X, original_x)


@pytest.mark.parametrize("missing", ["Age", "Occupation", "Credit_Score", "Customer_ID"])
def test_missing_required_columns_raise_clear_errors(missing: str) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        separate_features_target_groups(_cleaned_frame().drop(columns=missing))


def test_empty_fit_and_unsafe_feature_columns_raise_clear_errors() -> None:
    inputs = separate_features_target_groups(_cleaned_frame())
    preprocessor = build_model_preprocessor()

    with pytest.raises(ValueError, match="empty DataFrame"):
        preprocessor.fit(inputs.X.iloc[0:0])
    with pytest.raises(ValueError, match="Forbidden"):
        preprocessor.fit(inputs.X.assign(ID="ROW"))
    with pytest.raises(ValueError, match="Quarantined"):
        preprocessor.fit(inputs.X.assign(Credit_Mix="Good"))


def test_raw_source_file_is_never_modified(tmp_path: Path) -> None:
    raw_file = tmp_path / "train.csv"
    raw_file.write_bytes(b"Customer_ID,Credit_Score\nCUS_1,Good\n")
    before = hashlib.sha256(raw_file.read_bytes()).hexdigest()

    preprocessor, inputs = _fitted()
    preprocessor.transform(inputs.X)

    after = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    assert after == before
