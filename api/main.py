"""
FastAPI Application — Credit Scoring System.

Entry point for the REST API. Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src.config import API_TITLE, API_VERSION, API_DESCRIPTION, API_HOST, API_PORT
from src.logger import logger


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
)

# ──────────────────────────────────────────────
# CORS Middleware (allow Streamlit & frontends)
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Include Routes
# ──────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ──────────────────────────────────────────────
# Startup / Shutdown Events
# ──────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info(f" {API_TITLE} v{API_VERSION} starting on {API_HOST}:{API_PORT}")
    logger.info(" Swagger docs available at /docs")
    logger.info(" ReDoc available at /redoc")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(" Credit Scoring API shutting down.")


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
