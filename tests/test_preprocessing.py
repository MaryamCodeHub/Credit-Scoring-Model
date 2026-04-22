"""
Tests for the preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import engineer_features, encode_categoricals, reorder_features
from src.config import FINAL_FEATURE_ORDER


# ──────────────────────────────────────────────
# Test Data Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_input():
    """Create a sample raw input DataFrame."""
    return pd.DataFrame([{
        "Age": 30,
        "Gender": "Male",
        "Income": 75000,
        "Education": "Bachelor's Degree",
        "Marital Status": "Single",
        "Number of Children": 0,
        "Home Ownership": "Rented",
    }])


@pytest.fixture
def batch_input():
    """Create a batch of sample inputs."""
    return pd.DataFrame([
        {
            "Age": 25, "Gender": "Female", "Income": 50000,
            "Education": "High School Diploma", "Marital Status": "Single",
            "Number of Children": 0, "Home Ownership": "Rented",
        },
        {
            "Age": 45, "Gender": "Male", "Income": 120000,
            "Education": "Master's Degree", "Marital Status": "Married",
            "Number of Children": 2, "Home Ownership": "Owned",
        },
    ])


# ──────────────────────────────────────────────
# Feature Engineering Tests
# ──────────────────────────────────────────────

class TestFeatureEngineering:
    def test_income_per_dependent_no_children(self, sample_input):
        """Income_per_Dependent should equal Income when children=0."""
        result = engineer_features(sample_input)
        assert result["Income_per_Dependent"].iloc[0] == 75000.0  # 75000 / (0+1)

    def test_income_per_dependent_with_children(self):
        """Income_per_Dependent should divide by (children + 1)."""
        df = pd.DataFrame([{"Age": 40, "Income": 100000, "Number of Children": 3}])
        result = engineer_features(df)
        assert result["Income_per_Dependent"].iloc[0] == 25000.0  # 100000 / (3+1)

    def test_age_income_ratio(self, sample_input):
        """Age_Income_Ratio should be Income / Age."""
        result = engineer_features(sample_input)
        assert result["Age_Income_Ratio"].iloc[0] == 2500.0  # 75000 / 30

    def test_original_columns_preserved(self, sample_input):
        """Original columns should not be dropped."""
        result = engineer_features(sample_input)
        assert "Age" in result.columns
        assert "Income" in result.columns


# ──────────────────────────────────────────────
# Categorical Encoding Tests
# ──────────────────────────────────────────────

class TestCategoricalEncoding:
    def test_gender_encoding(self, sample_input):
        result = encode_categoricals(sample_input)
        assert result["Gender"].iloc[0] == 1  # Male = 1

    def test_education_ordinal(self, sample_input):
        result = encode_categoricals(sample_input)
        assert result["Education_Level"].iloc[0] == 3  # Bachelor's = 3

    def test_marital_status_encoding(self, sample_input):
        result = encode_categoricals(sample_input)
        assert result["Marital_Status"].iloc[0] == 0  # Single = 0

    def test_home_ownership_encoding(self, sample_input):
        result = encode_categoricals(sample_input)
        assert result["Home_Ownership"].iloc[0] == 0  # Rented = 0

    def test_original_categorical_dropped(self, sample_input):
        """Original string columns should be replaced by encoded versions."""
        result = encode_categoricals(sample_input)
        assert "Education" not in result.columns
        assert "Marital Status" not in result.columns
        assert "Home Ownership" not in result.columns

    def test_batch_encoding(self, batch_input):
        """Encoding should work on multiple rows."""
        result = encode_categoricals(batch_input)
        assert len(result) == 2
        assert result["Gender"].iloc[0] == 0  # Female
        assert result["Gender"].iloc[1] == 1  # Male


# ──────────────────────────────────────────────
# Feature Reordering Tests
# ──────────────────────────────────────────────

class TestFeatureReordering:
    def test_correct_order(self, sample_input):
        """Output columns should match FINAL_FEATURE_ORDER exactly."""
        df = engineer_features(sample_input)
        df = encode_categoricals(df)
        result = reorder_features(df)
        assert list(result.columns) == FINAL_FEATURE_ORDER

    def test_missing_feature_raises_error(self):
        """Should raise KeyError if a required feature is missing."""
        df = pd.DataFrame([{"Age": 30, "Gender": 1}])
        with pytest.raises(KeyError):
            reorder_features(df)
