"""Focused tests for verified model-v2 loading and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

import src.predict_v2 as predict_v2
from src.config import MODEL_V2_METADATA_PATH, MODEL_V2_PATH
from src.model_preprocessing import MODEL_FEATURES
from src.predict_v2 import (
    CreditScoreV2Service,
    ModelV2CompatibilityError,
    ModelV2IntegrityError,
    ModelV2UnavailableError,
)


def model_features(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "Age": 35,
        "Annual_Income": 72000.0,
        "Monthly_Inhand_Salary": 5000.0,
        "Num_Bank_Accounts": 4,
        "Num_Credit_Card": 4,
        "Num_of_Loan": 2,
        "Num_of_Delayed_Payment": 1,
        "Changed_Credit_Limit": 2.5,
        "Num_Credit_Inquiries": 2,
        "Outstanding_Debt": 850.0,
        "Credit_Utilization_Ratio": 28.0,
        "Credit_History_Age_Months": 120,
        "Total_EMI_per_month": 250.0,
        "Occupation": "Engineer",
    }
    values.update(overrides)
    return values


@pytest.fixture(scope="module")
def service() -> CreditScoreV2Service:
    return CreditScoreV2Service()


def test_verified_artifact_contract(service: CreditScoreV2Service) -> None:
    assert service.model_version == "2.0.0"
    assert list(MODEL_FEATURES) == list(service.metadata["raw_model_features"])
    assert service.metadata["transformed_feature_count"] == 35
    assert set(service.classifier.classes_) == {"Good", "Poor", "Standard"}


def test_predict_uses_actual_classifier_class_order(
    service: CreditScoreV2Service,
) -> None:
    result = service.predict(model_features())
    probabilities = result["probabilities"]
    assert result["credit_score"] in {"Poor", "Standard", "Good"}
    assert result["risk_level"] in {"High Risk", "Medium Risk", "Low Risk"}
    assert result["model_version"] == "2.0.0"
    assert np.isclose(sum(probabilities.values()), 1.0)
    predicted_key = str(result["credit_score"]).lower()
    assert result["confidence"] == probabilities[predicted_key]


def test_optional_missing_values_are_imputed_by_frozen_pipeline(
    service: CreditScoreV2Service,
) -> None:
    result = service.predict(
        model_features(
            Monthly_Inhand_Salary=None,
            Changed_Credit_Limit=None,
            Num_Credit_Inquiries=None,
            Occupation=None,
        )
    )
    assert result["credit_score"] in {"Poor", "Standard", "Good"}


def test_audited_extreme_values_are_not_rejected_or_clipped_by_service(
    service: CreditScoreV2Service,
) -> None:
    result = service.predict(
        model_features(
            Annual_Income=300000.0,
            Num_Bank_Accounts=12,
            Num_Credit_Card=12,
            Num_of_Loan=10,
            Num_of_Delayed_Payment=29,
            Num_Credit_Inquiries=18,
        )
    )
    assert result["credit_score"] in {"Poor", "Standard", "Good"}


def test_hash_is_verified_before_joblib_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = json.loads(MODEL_V2_METADATA_PATH.read_text(encoding="utf-8"))
    metadata["model_artifact_sha256"] = "0" * 64
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    load_called = False

    def forbidden_load(path: Path) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("joblib.load must not run before hash verification")

    monkeypatch.setattr(predict_v2.joblib, "load", forbidden_load)
    with pytest.raises(ModelV2IntegrityError):
        CreditScoreV2Service(MODEL_V2_PATH, metadata_path)
    assert load_called is False


def test_missing_artifact_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ModelV2UnavailableError, match="artifact"):
        CreditScoreV2Service(tmp_path / "missing.joblib", MODEL_V2_METADATA_PATH)


@pytest.mark.parametrize(
    ("artifact_value", "use_joblib"),
    [
        pytest.param(b"not a joblib artifact", False, id="corrupt-artifact"),
        pytest.param({"unexpected": "structure"}, True, id="invalid-structure"),
    ],
)
def test_corrupt_or_structurally_invalid_artifact_is_rejected(
    tmp_path: Path,
    artifact_value: object,
    use_joblib: bool,
) -> None:
    artifact_path = tmp_path / "model.joblib"
    if use_joblib:
        joblib.dump(artifact_value, artifact_path)
    else:
        artifact_path.write_bytes(artifact_value)

    metadata = json.loads(MODEL_V2_METADATA_PATH.read_text(encoding="utf-8"))
    metadata["model_artifact_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelV2CompatibilityError):
        CreditScoreV2Service(artifact_path, metadata_path)


def test_predict_does_not_call_fit(
    service: CreditScoreV2Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("fit must never be called during inference")

    monkeypatch.setattr(service.pipeline, "fit", forbidden_fit)
    service.predict(model_features())
