"""API-contract and backward-compatibility tests for model v2."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

import api.main as main_module
from api.main import app
from api.schemas_v2 import CreditScoreV2Request
from src.model_preprocessing import MODEL_FEATURES
from src.predict_v2 import (
    ModelV2CompatibilityError,
    ModelV2InferenceError,
    ModelV2IntegrityError,
    ModelV2UnavailableError,
)


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "age": 35,
        "annual_income": 72000.0,
        "monthly_inhand_salary": 5000.0,
        "num_bank_accounts": 4,
        "num_credit_cards": 4,
        "num_loans": 2,
        "num_delayed_payments": 1,
        "changed_credit_limit": 2.5,
        "num_credit_inquiries": 2,
        "outstanding_debt": 850.0,
        "credit_utilization_ratio": 28.0,
        "credit_history_age_months": 120,
        "total_emi_per_month": 250.0,
        "occupation": "Engineer",
    }
    payload.update(overrides)
    return payload


class StubService:
    def __init__(self) -> None:
        self.features: dict[str, object] | None = None
        self.model_version = "2.0.0"

    def predict(self, features: dict[str, object]) -> dict[str, object]:
        self.features = features
        return {
            "credit_score": "Standard",
            "probabilities": {"poor": 0.21, "standard": 0.64, "good": 0.15},
            "confidence": 0.64,
            "risk_level": "Medium Risk",
            "model_version": "2.0.0",
        }


@pytest.fixture
def client_and_service(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, StubService]]:
    service = StubService()
    monkeypatch.setattr(main_module, "load_v2_service", lambda: service)
    with TestClient(app) as client:
        yield client, service


def test_v2_response_and_exact_feature_mapping(
    client_and_service: tuple[TestClient, StubService],
) -> None:
    client, service = client_and_service
    response = client.post("/api/v2/predict", json=valid_payload())
    assert response.status_code == 200
    assert response.json() == {
        "credit_score": "Standard",
        "probabilities": {"poor": 0.21, "standard": 0.64, "good": 0.15},
        "confidence": 0.64,
        "risk_level": "Medium Risk",
        "model_version": "2.0.0",
    }
    assert service.features is not None
    assert list(service.features) == list(MODEL_FEATURES)
    assert service.features["Num_Credit_Card"] == 4
    assert service.features["Credit_History_Age_Months"] == 120


def test_optional_fields_are_nullable(
    client_and_service: tuple[TestClient, StubService],
) -> None:
    client, service = client_and_service
    payload = valid_payload()
    for field in (
        "monthly_inhand_salary",
        "changed_credit_limit",
        "num_credit_inquiries",
        "occupation",
    ):
        payload.pop(field)
    response = client.post("/api/v2/predict", json=payload)
    assert response.status_code == 200
    assert service.features is not None
    assert service.features["Monthly_Inhand_Salary"] is None
    assert service.features["Occupation"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 17),
        ("age", 121),
        ("num_loans", -1),
        ("num_bank_accounts", 1.5),
        ("annual_income", -1),
        ("outstanding_debt", -1),
        ("credit_utilization_ratio", 100.1),
        ("credit_history_age_months", -1),
        ("total_emi_per_month", -1),
        ("occupation", "   "),
        ("occupation", "Engineer\nManager"),
    ],
)
def test_invalid_values_are_rejected(
    client_and_service: tuple[TestClient, StubService],
    field: str,
    value: object,
) -> None:
    client, _ = client_and_service
    response = client.post("/api/v2/predict", json=valid_payload(**{field: value}))
    assert response.status_code == 422


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        CreditScoreV2Request.model_validate(
            valid_payload(changed_credit_limit=value)
        )


def test_extra_and_missing_fields_are_rejected(
    client_and_service: tuple[TestClient, StubService],
) -> None:
    client, _ = client_and_service
    assert client.post(
        "/api/v2/predict", json=valid_payload(unexpected=1)
    ).status_code == 422
    payload = valid_payload()
    payload.pop("annual_income")
    assert client.post("/api/v2/predict", json=payload).status_code == 422


def test_extreme_values_are_passed_to_frozen_pipeline(
    client_and_service: tuple[TestClient, StubService],
) -> None:
    client, service = client_and_service
    response = client.post(
        "/api/v2/predict",
        json=valid_payload(
            annual_income=300000.0,
            num_bank_accounts=12,
            num_credit_cards=12,
            num_loans=10,
            num_delayed_payments=29,
            num_credit_inquiries=18,
        ),
    )
    assert response.status_code == 200
    assert service.features is not None
    assert service.features["Annual_Income"] == 300000.0
    assert service.features["Num_Bank_Accounts"] == 12
    assert service.features["Num_Credit_Inquiries"] == 18


def test_occupation_is_trimmed(
    client_and_service: tuple[TestClient, StubService],
) -> None:
    client, service = client_and_service
    assert client.post(
        "/api/v2/predict", json=valid_payload(occupation="  Engineer  ")
    ).status_code == 200
    assert service.features is not None
    assert service.features["Occupation"] == "Engineer"


def test_inference_error_does_not_expose_internal_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingService:
        def predict(self, features: dict[str, object]) -> dict[str, object]:
            raise ModelV2InferenceError("secret internal path")

    service = FailingService()
    service.model_version = "2.0.0"
    monkeypatch.setattr(main_module, "load_v2_service", lambda: service)
    with TestClient(app) as client:
        response = client.post("/api/v2/predict", json=valid_payload())
    assert response.status_code == 500
    assert "secret internal path" not in response.text


def test_lifespan_loads_v2_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = StubService()
    load_count = 0

    def counted_loader() -> StubService:
        nonlocal load_count
        load_count += 1
        return service

    monkeypatch.setattr(main_module, "load_v2_service", counted_loader)
    with TestClient(app) as client:
        assert client.get("/api/v2/health").status_code == 200
        assert client.get("/api/v2/readiness").status_code == 200
        assert client.post("/api/v2/predict", json=valid_payload()).status_code == 200
        assert client.post("/api/v2/predict", json=valid_payload()).status_code == 200
        assert app.state.v2_service is service
    assert load_count == 1


def test_ready_health_and_readiness_responses(
    client_and_service: tuple[TestClient, StubService],
) -> None:
    client, _ = client_and_service
    assert client.get("/api/v2/health").json() == {
        "status": "healthy",
        "api_version": "v2",
    }
    readiness = client.get("/api/v2/readiness")
    assert readiness.status_code == 200
    assert readiness.json() == {
        "status": "ready",
        "model_ready": True,
        "model_version": "2.0.0",
        "artifact_integrity": "verified",
    }


@pytest.mark.parametrize(
    "startup_error",
    [
        pytest.param(
            ModelV2UnavailableError(r"C:\private\models\missing.joblib"),
            id="missing-artifact",
        ),
        pytest.param(
            ModelV2IntegrityError("secret hash mismatch"),
            id="hash-mismatch",
        ),
        pytest.param(
            ModelV2CompatibilityError("secret corrupt artifact"),
            id="corrupt-artifact",
        ),
        pytest.param(
            ModelV2CompatibilityError("secret invalid pipeline structure"),
            id="invalid-structure",
        ),
    ],
)
def test_artifact_failures_degrade_v2_and_prediction_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    startup_error: Exception,
) -> None:
    def failing_loader() -> StubService:
        raise startup_error

    monkeypatch.setattr(main_module, "load_v2_service", failing_loader)
    with TestClient(app) as client:
        assert client.get("/api/v2/health").status_code == 200
        readiness = client.get("/api/v2/readiness")
        prediction = client.post("/api/v2/predict", json=valid_payload())

    assert readiness.status_code == 503
    assert readiness.json() == {
        "status": "degraded",
        "model_ready": False,
        "model_version": None,
        "artifact_integrity": "failed",
    }
    assert prediction.status_code == 503
    assert prediction.json() == {"detail": "Credit model v2 is unavailable."}
    public_text = readiness.text + prediction.text
    assert "private" not in public_text
    assert "secret" not in public_text
    assert "joblib" not in public_text


def test_unexpected_inference_failure_is_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedFailureService(StubService):
        def predict(self, features: dict[str, object]) -> dict[str, object]:
            raise RuntimeError(r"C:\private\models\internal failure")

    monkeypatch.setattr(
        main_module, "load_v2_service", lambda: UnexpectedFailureService()
    )
    with TestClient(app) as client:
        response = client.post("/api/v2/predict", json=valid_payload())
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Credit model v2 could not produce a prediction."
    }
    assert "private" not in response.text
    assert "internal failure" not in response.text


def test_legacy_v1_health_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "load_v2_service", StubService)
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_cors_does_not_allow_credentials_with_wildcard_origins() -> None:
    cors = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )
    assert not (
        cors.kwargs.get("allow_origins") == ["*"]
        and cors.kwargs.get("allow_credentials") is True
    )


def test_lifespan_emits_no_fastapi_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "load_v2_service", StubService)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TestClient(app) as client:
            assert client.get("/api/v1/health").status_code == 200
    lifecycle_warnings = [
        warning
        for warning in caught
        if "on_event is deprecated" in str(warning.message)
    ]
    assert lifecycle_warnings == []
