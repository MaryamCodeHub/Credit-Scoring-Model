"""
API Routes — Credit Scoring System.

Defines the /predict and /health endpoints.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import CreditScoreRequest, CreditScoreResponse, HealthResponse
from src.predict import CreditScorer
from src.config import API_VERSION
from src.logger import logger

router = APIRouter()

# ──────────────────────────────────────────────
# Initialize the scorer (loaded once at startup)
# ──────────────────────────────────────────────
try:
    scorer = CreditScorer()
except FileNotFoundError:
    scorer = None
    logger.warning(
        "Model artifacts not found. /predict will return 503. "
        "Place model files in models/ and restart."
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=CreditScoreResponse,
    summary="Predict Credit Score",
    description=(
        "Submit applicant data and receive a credit score prediction "
        "with confidence probability and risk assessment."
    ),
    tags=["Prediction"],
)
async def predict_credit_score(request: CreditScoreRequest):
    """
    Predict the credit score for a loan applicant.

    Accepts demographic and financial data, runs it through
    the preprocessing pipeline, and returns the model's prediction.
    """
    if scorer is None or not scorer.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not available. Please ensure model artifacts "
                "(credit_model.pkl, scaler.pkl, target_encoder.pkl) "
                "are placed in the models/ directory."
            ),
        )

    try:
        # Convert Pydantic model to dict with original column names
        input_data = {
            "Age": request.Age,
            "Gender": request.Gender.value,
            "Income": request.Income,
            "Education": request.Education.value,
            "Marital Status": request.Marital_Status.value,
            "Number of Children": request.Number_of_Children,
            "Home Ownership": request.Home_Ownership.value,
        }

        result = scorer.predict(input_data)
        return CreditScoreResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and the model is loaded.",
    tags=["System"],
)
async def health_check():
    """Return service health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=scorer is not None and scorer.is_loaded,
        version=API_VERSION,
    )
