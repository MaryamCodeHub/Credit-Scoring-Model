"""FastAPI routes for the integrity-checked v2 inference service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas_v2 import CreditScoreV2Request, CreditScoreV2Response
from src.predict_v2 import CreditScoreV2Service, ModelV2Error


router = APIRouter()


def get_v2_service(request: Request) -> CreditScoreV2Service:
    service = getattr(request.app.state, "v2_service", None)
    ready = getattr(request.app.state, "v2_model_ready", False)
    if not ready or service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credit model v2 is unavailable.",
        )
    return service


@router.get("/health", summary="Check whether the v2 API process is healthy")
def health_v2() -> dict[str, str]:
    return {"status": "healthy", "api_version": "v2"}


@router.get("/readiness", summary="Check verified model-v2 readiness")
def readiness_v2(request: Request) -> JSONResponse:
    ready = bool(getattr(request.app.state, "v2_model_ready", False))
    content = {
        "status": "ready" if ready else "degraded",
        "model_ready": ready,
        "model_version": getattr(request.app.state, "v2_model_version", None),
        "artifact_integrity": getattr(
            request.app.state, "v2_artifact_integrity", "failed"
        ),
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK
        if ready
        else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=content,
    )


@router.post(
    "/predict",
    response_model=CreditScoreV2Response,
    summary="Predict a credit-score class with verified model v2",
)
def predict_credit_score_v2(
    request: CreditScoreV2Request,
    service: CreditScoreV2Service = Depends(get_v2_service),
) -> CreditScoreV2Response:
    try:
        result = service.predict(request.to_model_features())
        return CreditScoreV2Response.model_validate(result)
    except ModelV2Error:
        logger.exception("Model v2 inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Credit model v2 could not produce a prediction.",
        ) from None
    except Exception:
        logger.exception("Unexpected model v2 inference failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Credit model v2 could not produce a prediction.",
        ) from None
