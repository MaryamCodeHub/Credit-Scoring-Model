# ════════════════════════════════════════════════════════════════
# Credit Scoring System — Multi-Stage Production Dockerfile
# ════════════════════════════════════════════════════════════════
# Build:  docker build -t credit-scoring-api .
# Run:    docker run -p 8000:8000 credit-scoring-api
# ════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Install dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY data/ ./data/

# ── Health check ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); assert r.status_code == 200"

# ── Expose ports ───────────────────────────────────────────────
# 8000 = FastAPI  |  8501 = Streamlit
EXPOSE 8000 8501

# ── Default: Run FastAPI ───────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
