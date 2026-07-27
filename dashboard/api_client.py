"""Safe, typed HTTP client for the FastAPI v2 dashboard boundary."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

import httpx


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
HEALTH_ENDPOINT = "/api/v2/health"
READINESS_ENDPOINT = "/api/v2/readiness"
PREDICTION_ENDPOINT = "/api/v2/predict"
EXPECTED_CLASSES = ("poor", "standard", "good")


@dataclass(frozen=True)
class ApiStatus:
    available: bool
    ready: bool
    message: str
    status_code: int | None = None
    model_version: str | None = None
    artifact_integrity: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    ok: bool
    message: str
    status_code: int | None = None
    credit_score: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    risk_level: str | None = None
    model_version: str | None = None


class CreditApiClient:
    """Make bounded, non-retrying requests to the configured v2 API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        connect_timeout: float = 2.0,
        read_timeout: float = 8.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        configured = base_url or os.environ.get(
            "CREDIT_API_BASE_URL", DEFAULT_API_BASE_URL
        )
        self.base_url = configured.rstrip("/")
        self.timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )
        self.transport = transport

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_payload: Mapping[str, object] | None = None,
    ) -> tuple[httpx.Response | None, str | None]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = client.request(method, endpoint, json=json_payload)
        except httpx.TimeoutException:
            return None, "The API request timed out. Please try again later."
        except httpx.RequestError:
            return None, "The API is currently unavailable."
        return response, None

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
        try:
            payload = response.json()
        except (ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def health(self) -> ApiStatus:
        response, error = self._request("GET", HEALTH_ENDPOINT)
        if error:
            return ApiStatus(False, False, error)
        payload = self._safe_json(response)
        if response.status_code != 200 or payload is None:
            return ApiStatus(
                True,
                False,
                "The API health response was not valid.",
                response.status_code,
            )
        return ApiStatus(
            True,
            False,
            "The API process is healthy.",
            response.status_code,
        )

    def readiness(self) -> ApiStatus:
        response, error = self._request("GET", READINESS_ENDPOINT)
        if error:
            return ApiStatus(False, False, error)
        payload = self._safe_json(response)
        if payload is None:
            return ApiStatus(
                True,
                False,
                "The API readiness response was not valid.",
                response.status_code,
            )
        ready = (
            response.status_code == 200
            and payload.get("model_ready") is True
            and payload.get("artifact_integrity") == "verified"
        )
        return ApiStatus(
            available=True,
            ready=ready,
            message=(
                "Model v2 is verified and ready."
                if ready
                else "Model v2 is not currently ready."
            ),
            status_code=response.status_code,
            model_version=(
                payload.get("model_version")
                if isinstance(payload.get("model_version"), str)
                else None
            ),
            artifact_integrity=(
                payload.get("artifact_integrity")
                if isinstance(payload.get("artifact_integrity"), str)
                else None
            ),
        )

    def predict(self, payload: Mapping[str, object]) -> PredictionResult:
        response, error = self._request(
            "POST", PREDICTION_ENDPOINT, json_payload=payload
        )
        if error:
            return PredictionResult(False, error)
        if response.status_code == 422:
            return PredictionResult(
                False,
                "Some inputs were not accepted. Review the highlighted values.",
                422,
            )
        if response.status_code == 503:
            return PredictionResult(
                False,
                "Model v2 is temporarily unavailable.",
                503,
            )
        if response.status_code >= 500:
            return PredictionResult(
                False,
                "The prediction service encountered an unexpected error.",
                response.status_code,
            )
        if response.status_code != 200:
            return PredictionResult(
                False,
                "The prediction request could not be completed.",
                response.status_code,
            )

        body = self._safe_json(response)
        parsed = self._parse_prediction(body)
        if parsed is None:
            return PredictionResult(
                False,
                "The API returned a malformed prediction response.",
                response.status_code,
            )
        return parsed

    @staticmethod
    def _parse_prediction(body: dict[str, Any] | None) -> PredictionResult | None:
        if body is None:
            return None
        score = body.get("credit_score")
        risk = body.get("risk_level")
        version = body.get("model_version")
        confidence = body.get("confidence")
        probabilities = body.get("probabilities")
        if (
            score not in {"Poor", "Standard", "Good"}
            or risk not in {"High Risk", "Medium Risk", "Low Risk"}
            or version != "2.0.0"
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not isinstance(probabilities, dict)
            or set(probabilities) != set(EXPECTED_CLASSES)
        ):
            return None
        try:
            parsed_probabilities = {
                key: float(probabilities[key]) for key in EXPECTED_CLASSES
            }
        except (TypeError, ValueError):
            return None
        values = list(parsed_probabilities.values())
        if (
            not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-8)
            or not 0 <= float(confidence) <= 1
            or not math.isclose(
                float(confidence),
                parsed_probabilities[score.lower()],
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            return None
        return PredictionResult(
            ok=True,
            message="Prediction completed.",
            status_code=200,
            credit_score=score,
            probabilities=parsed_probabilities,
            confidence=float(confidence),
            risk_level=risk,
            model_version=version,
        )
