"""
Central Configuration — Credit Scoring System.

All paths, constants, feature definitions, and model parameters
are defined here. No magic strings scattered across the codebase.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# ──────────────────────────────────────────────
# Data Files
# ──────────────────────────────────────────────
RAW_DATA_PATH = DATA_DIR / "Credit_Score_Classification_Dataset.csv"

# ──────────────────────────────────────────────
# Model Artifacts (produced by Colab training)
# ──────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "credit_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
MODEL_V2_DIR = MODELS_DIR / "v2"
MODEL_V2_PATH = MODEL_V2_DIR / "credit_score_decision_tree_pipeline.joblib"
MODEL_V2_METADATA_PATH = MODEL_V2_DIR / "model_metadata.json"

# ──────────────────────────────────────────────
# Feature Definitions
# ──────────────────────────────────────────────

# Exact column names from the Kaggle CSV
RAW_FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Income",
    "Education",
    "Marital Status",
    "Number of Children",
    "Home Ownership",
]
TARGET_COLUMN = "Credit Score"

# Ordinal mapping for Education (preserves natural hierarchy)
EDUCATION_ORDER = {
    "Intermediate": 1,
    "Associate's Degree": 2,
    "Bachelor's Degree": 3,
    "Master's Degree": 4,
    "Doctorate": 5,
}

# Binary mappings
GENDER_MAP = {"Female": 0, "Male": 1}
MARITAL_STATUS_MAP = {"Single": 0, "Married": 1}
HOME_OWNERSHIP_MAP = {"Rented": 0, "Owned": 1}

# Numerical features to scale (after engineering)
NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "Number_of_Children",
    "Income_per_Dependent",
    "Age_Income_Ratio",
]

# Final feature order for model input (must match Colab training)
FINAL_FEATURE_ORDER = [
    "Age",
    "Gender",
    "Income",
    "Education_Level",
    "Marital_Status",
    "Number_of_Children",
    "Home_Ownership",
    "Income_per_Dependent",
    "Age_Income_Ratio",
]

# ──────────────────────────────────────────────
# Model Training Defaults
# ──────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# API Settings
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Credit Scoring API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-grade REST API for real-time credit score prediction. "
    "Accepts applicant financial and demographic data, returns credit score "
    "classification (Low / Average / High) with confidence probabilities."
)

# ──────────────────────────────────────────────
# Dashboard Settings
# ──────────────────────────────────────────────
DASHBOARD_TITLE = "🏦 Credit Scoring Dashboard"
THEME_PRIMARY = "#009688"       # Vivid Teal
THEME_SECONDARY = "#66BB6A"     # Mint Green
THEME_BACKGROUND = "#0D1117"    # Dark background
THEME_SURFACE = "#161B22"       # Card surface
THEME_TEXT = "#E6EDF3"          # Light text
