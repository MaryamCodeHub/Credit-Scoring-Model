"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# Test Client Setup
# ──────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a test client for the API."""
    from api.main import app
    return TestClient(app)


# ──────────────────────────────────────────────
# Health Check Tests
# ──────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


# ──────────────────────────────────────────────
# Root Endpoint Tests
# ──────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_docs_link(self, client):
        response = client.get("/")
        data = response.json()
        assert "docs" in data


# ──────────────────────────────────────────────
# Predict Endpoint Tests (validation only — no model needed)
# ──────────────────────────────────────────────

class TestPredictValidation:
    def test_predict_rejects_missing_fields(self, client):
        """Should return 422 for incomplete input."""
        response = client.post("/api/v1/predict", json={"Age": 30})
        assert response.status_code == 422

    def test_predict_rejects_invalid_age(self, client):
        """Should reject age below 18."""
        response = client.post(
            "/api/v1/predict",
            json={
                "Age": 10,
                "Gender": "Male",
                "Income": 50000,
                "Education": "Bachelor's Degree",
                "Marital Status": "Single",
                "Number of Children": 0,
                "Home Ownership": "Rented",
            },
        )
        assert response.status_code == 422

    def test_predict_rejects_invalid_gender(self, client):
        """Should reject invalid gender value."""
        response = client.post(
            "/api/v1/predict",
            json={
                "Age": 30,
                "Gender": "Other",
                "Income": 50000,
                "Education": "Bachelor's Degree",
                "Marital Status": "Single",
                "Number of Children": 0,
                "Home Ownership": "Rented",
            },
        )
        assert response.status_code == 422

    def test_predict_rejects_negative_income(self, client):
        """Should reject income <= 0."""
        response = client.post(
            "/api/v1/predict",
            json={
                "Age": 30,
                "Gender": "Male",
                "Income": -5000,
                "Education": "Bachelor's Degree",
                "Marital Status": "Single",
                "Number of Children": 0,
                "Home Ownership": "Rented",
            },
        )
        assert response.status_code == 422
