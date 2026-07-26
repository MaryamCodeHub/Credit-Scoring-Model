"""
FastAPI Application — Credit Scoring System.

Entry point for the REST API. Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.routes_v2 import router as router_v2
from src.config import API_TITLE, API_VERSION, API_DESCRIPTION, API_HOST, API_PORT
from src.logger import logger
from src.predict_v2 import CreditScoreV2Service, ModelV2Error


def load_v2_service() -> CreditScoreV2Service:
    """Construct the verified v2 service once during application startup."""
    return CreditScoreV2Service()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize v2 safely without making an artifact failure fatal to v1."""
    application.state.v2_service = None
    application.state.v2_model_ready = False
    application.state.v2_model_version = None
    application.state.v2_artifact_integrity = "failed"

    try:
        service = load_v2_service()
    except ModelV2Error:
        logger.exception("Model v2 startup verification failed; v2 is degraded.")
    except Exception:
        logger.exception("Unexpected model v2 startup failure; v2 is degraded.")
    else:
        application.state.v2_service = service
        application.state.v2_model_ready = True
        application.state.v2_model_version = service.model_version
        application.state.v2_artifact_integrity = "verified"

    logger.info(f" {API_TITLE} v{API_VERSION} starting on {API_HOST}:{API_PORT}")
    logger.info(" Swagger docs available at /docs")
    logger.info(" ReDoc available at /redoc")
    yield
    logger.info(" Credit Scoring API shutting down.")


# ──────────────────────────────────────────────
# Create FastAPI Application
# ──────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# CORS Middleware (allow Streamlit & frontends)
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Include Routes
# ──────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")
app.include_router(router_v2, prefix="/api/v2")


# ──────────────────────────────────────────────
# Startup / Shutdown Events
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Root Redirect
# ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": f"Welcome to {API_TITLE}",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict (POST)",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
