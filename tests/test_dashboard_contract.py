"""Focused safety and presentation contracts for the Phase 4D dashboard."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import httpx
import pytest

from dashboard.api_client import (
    PREDICTION_ENDPOINT,
    READINESS_ENDPOINT,
    ApiStatus,
    CreditApiClient,
)
from dashboard.components import (
    CLASS_ORDER,
    CONFUSION_MATRIX,
    EDUCATIONAL_DISCLAIMER,
    FINAL_METRICS,
    NAVIGATION_SECTIONS,
    OPTIONAL_PREDICTION_FIELDS,
    PARTITION_SIZES,
    PER_CLASS_METRICS,
    PREDICTION_FIELDS,
    VALIDATION_MACRO_F1,
    AnalyticsIntegrityError,
    build_prediction_payload,
    confusion_matrix_figure,
    load_model_metadata,
    load_verified_analytics,
    per_class_metrics_figure,
    prediction_is_available,
    target_distribution_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def _source_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(DASHBOARD_DIR.glob("*.py"))
    )


def _payload() -> dict[str, object]:
    return {
        "age": 35,
        "annual_income": 72000.0,
        "monthly_inhand_salary": None,
        "num_bank_accounts": 4,
        "num_credit_cards": 3,
        "num_loans": 2,
        "num_delayed_payments": 1,
        "changed_credit_limit": None,
        "num_credit_inquiries": None,
        "outstanding_debt": 1250.0,
        "credit_utilization_ratio": 28.5,
        "credit_history_age_months": 132,
        "total_emi_per_month": 210.0,
        "occupation": None,
    }


def _client(handler) -> CreditApiClient:
    return CreditApiClient(
        "http://api.example.test",
        transport=httpx.MockTransport(handler),
    )


def test_dashboard_has_no_model_or_raw_data_access() -> None:
    source = _source_text().lower()
    forbidden = (
        "import joblib",
        "from joblib",
        "read_csv",
        ".csv",
        "data/raw",
        "kaggle_credit_score",
        "creditscorer",
        "clean_credit_data",
    )
    assert all(token not in source for token in forbidden)


def test_dashboard_uses_only_v2_api_endpoints() -> None:
    source = (DASHBOARD_DIR / "api_client.py").read_text(encoding="utf-8")
    assert "/api/v1" not in source
    assert "/api/v2/health" in source
    assert READINESS_ENDPOINT in source
    assert PREDICTION_ENDPOINT in source


def test_exact_prediction_payload_and_optional_nulls() -> None:
    payload = build_prediction_payload(_payload())
    assert tuple(payload) == PREDICTION_FIELDS
    assert set(payload) == set(PREDICTION_FIELDS)
    assert all(payload[field] is None for field in OPTIONAL_PREDICTION_FIELDS)


def test_payload_contract_rejects_missing_or_extra_fields() -> None:
    values = _payload()
    values.pop("age")
    with pytest.raises(ValueError, match="missing"):
        build_prediction_payload(values)
    values = _payload() | {"unexpected": 1}
    with pytest.raises(ValueError, match="extra"):
        build_prediction_payload(values)


def test_readiness_controls_prediction_availability() -> None:
    assert prediction_is_available(ApiStatus(True, True, "ready"))
    assert not prediction_is_available(ApiStatus(True, False, "degraded"))
    assert not prediction_is_available(ApiStatus(False, False, "offline"))


def test_ready_response_is_parsed_from_real_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == READINESS_ENDPOINT
        return httpx.Response(
            200,
            json={
                "status": "ready",
                "model_ready": True,
                "model_version": "2.0.0",
                "artifact_integrity": "verified",
            },
        )

    result = _client(handler).readiness()
    assert result.available and result.ready
    assert result.model_version == "2.0.0"
    assert result.artifact_integrity == "verified"


@pytest.mark.parametrize(
    ("status_code", "expected_fragment"),
    [
        (422, "not accepted"),
        (503, "temporarily unavailable"),
        (500, "unexpected error"),
    ],
)
def test_prediction_http_errors_are_safe(
    status_code: int, expected_fragment: str
) -> None:
    internal_detail = r"C:\private\models\secret.joblib"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": internal_detail})

    result = _client(handler).predict(_payload())
    assert not result.ok
    assert expected_fragment in result.message
    assert internal_detail not in result.message
    assert "\\" not in result.message


def test_timeout_and_unavailable_errors_are_safe() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("internal timeout detail", request=request)

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private host detail", request=request)

    timed_out = _client(timeout).readiness()
    offline = _client(unavailable).readiness()
    assert timed_out.message == "The API request timed out. Please try again later."
    assert offline.message == "The API is currently unavailable."


def test_malformed_prediction_response_is_blocked() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "credit_score": "Standard",
                "probabilities": {"poor": 0.2, "standard": 0.6},
                "confidence": 0.6,
                "risk_level": "Medium Risk",
                "model_version": "2.0.0",
            },
        )

    result = _client(handler).predict(_payload())
    assert not result.ok
    assert result.message == "The API returned a malformed prediction response."


def test_valid_prediction_response_is_structured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == PREDICTION_ENDPOINT
        assert json.loads(request.content) == _payload()
        return httpx.Response(
            200,
            json={
                "credit_score": "Standard",
                "probabilities": {"poor": 0.2, "standard": 0.65, "good": 0.15},
                "confidence": 0.65,
                "risk_level": "Medium Risk",
                "model_version": "2.0.0",
            },
        )

    result = _client(handler).predict(_payload())
    assert result.ok
    assert result.credit_score == "Standard"
    assert result.risk_level == "Medium Risk"


def test_analytics_checksum_is_verified_before_json_parsing(
    tmp_path: Path,
) -> None:
    artifact = {
        "source_partition": "development_train",
        "privacy": {"aggregate_only": True, "suppression_threshold": 25},
    }
    payload = json.dumps(artifact).encode()
    json_path = tmp_path / "analytics.json"
    sidecar_path = tmp_path / "analytics.sha256"
    json_path.write_bytes(payload)
    sidecar_path.write_text(hashlib.sha256(payload).hexdigest(), encoding="ascii")
    assert load_verified_analytics(json_path, sidecar_path) == artifact


def test_checksum_mismatch_blocks_analytics(tmp_path: Path) -> None:
    json_path = tmp_path / "analytics.json"
    sidecar_path = tmp_path / "analytics.sha256"
    json_path.write_text("{}", encoding="utf-8")
    sidecar_path.write_text("0" * 64, encoding="ascii")
    with pytest.raises(AnalyticsIntegrityError, match="integrity"):
        load_verified_analytics(json_path, sidecar_path)


def test_committed_analytics_is_aggregate_development_only() -> None:
    analytics = load_verified_analytics()
    assert analytics["source_partition"] == "development_train"
    assert analytics["privacy"]["aggregate_only"] is True
    assert analytics["privacy"]["suppression_threshold"] == 25
    assert target_distribution_figure(analytics).data


def test_navigation_and_educational_boundary_are_explicit() -> None:
    assert NAVIGATION_SECTIONS == (
        "Executive Overview",
        "Credit Prediction",
        "Data Insights",
        "Model Performance",
        "Model Card & Limitations",
    )
    assert "Educational portfolio project only" in EDUCATIONAL_DISCLAIMER
    assert "not approved for real lending decisions" in EDUCATIONAL_DISCLAIMER


def test_no_fake_status_gauge_history_or_external_fonts() -> None:
    source = _source_text().lower()
    forbidden = (
        "api live",
        "gauge",
        "session_state",
        "recent activity",
        "fonts.googleapis.com",
        "linear-gradient",
        "radial-gradient",
        "glassmorphism",
        "production-ready",
        "ready for production",
    )
    assert all(token not in source for token in forbidden)


def test_fixed_audited_performance_values_and_class_order() -> None:
    assert VALIDATION_MACRO_F1 == 0.619834
    assert FINAL_METRICS == {
        "macro_f1": 0.6288978544,
        "accuracy": 0.6742021277,
        "balanced_accuracy": 0.6149644096,
        "weighted_f1": 0.6688847035,
        "macro_precision": 0.6533535038,
        "macro_recall": 0.6149644096,
        "poor_recall": 0.5757722008,
    }
    assert CLASS_ORDER == ("Poor", "Standard", "Good")
    assert CONFUSION_MATRIX == (
        (2386, 1445, 313),
        (832, 6039, 725),
        (24, 1316, 1208),
    )
    assert PER_CLASS_METRICS["Good"]["support"] == 2548
    assert PARTITION_SIZES["development"] == {
        "rows": 71424,
        "customers": 8928,
    }
    assert PARTITION_SIZES["validation"] == {
        "rows": 14288,
        "customers": 1786,
    }
    assert PARTITION_SIZES["final_test"] == {
        "rows": 14288,
        "customers": 1786,
    }


def test_model_metadata_matches_dashboard_card() -> None:
    metadata = load_model_metadata()
    assert metadata["model_version"] == "2.0.0"
    assert metadata["model_type"] == "DecisionTreeClassifier"
    assert len(metadata["raw_model_features"]) == 14
    assert metadata["transformed_feature_count"] == 35


def test_components_render_without_browser() -> None:
    assert len(per_class_metrics_figure().data) == 3
    matrix = confusion_matrix_figure()
    assert tuple(matrix.data[0].x) == CLASS_ORDER
    assert tuple(matrix.data[0].y) == CLASS_ORDER


def test_dashboard_import_does_not_contact_api(monkeypatch) -> None:
    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("Dashboard import must not contact the API.")

    monkeypatch.setattr(httpx.Client, "request", unexpected_request)
    module = importlib.import_module("dashboard.app")
    importlib.reload(module)


def test_app_uses_api_client_for_the_single_prediction_boundary() -> None:
    source = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8")
    assert source.count("client.predict(") == 1
    assert "CreditApiClient" in source
    assert ".fit(" not in source
    assert ".transform(" not in source
    assert "st.exception" not in source
