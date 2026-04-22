"""
Preprocessing Pipeline — Credit Scoring System.

This module mirrors the EXACT preprocessing steps used during
model training on Google Colab. Any change here MUST be reflected
in the Colab training notebook and vice versa.

Pipeline Steps:
    1. Feature Engineering (Income_per_Dependent, Age_Income_Ratio)
    2. Categorical Encoding (ordinal + binary maps)
    3. Column Renaming (spaces → underscores for ML compatibility)
    4. Feature Ordering (match training order)
    5. Scaling (StandardScaler on numerical features)
"""

import numpy as np
import pandas as pd
import joblib
from typing import Union

from src.config import (
    EDUCATION_ORDER,
    GENDER_MAP,
    MARITAL_STATUS_MAP,
    HOME_OWNERSHIP_MAP,
    NUMERICAL_FEATURES,
    FINAL_FEATURE_ORDER,
    SCALER_PATH,
)
from src.logger import logger


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw input data.

    New Features:
        - Income_per_Dependent: Income / (Number of Children + 1)
        - Age_Income_Ratio: Income / Age

    Args:
        df: DataFrame with raw features.

    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()

    # Income per dependent — captures financial burden
    children_col = "Number of Children" if "Number of Children" in df.columns else "Number_of_Children"
    df["Income_per_Dependent"] = df["Income"] / (df[children_col] + 1)

    # Age-Income ratio — captures career progression
    df["Age_Income_Ratio"] = df["Income"] / df["Age"]

    logger.debug(
        f"Engineered features: Income_per_Dependent range "
        f"[{df['Income_per_Dependent'].min():.0f}, {df['Income_per_Dependent'].max():.0f}], "
        f"Age_Income_Ratio range "
        f"[{df['Age_Income_Ratio'].min():.0f}, {df['Age_Income_Ratio'].max():.0f}]"
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using deterministic mappings.

    Encoding Strategy:
        - Education: Ordinal (1-5, preserving hierarchy)
        - Gender: Binary (Female=0, Male=1)
        - Marital Status: Binary (Single=0, Married=1)
        - Home Ownership: Binary (Rented=0, Owned=1)

    Args:
        df: DataFrame with raw categorical columns.

    Returns:
        DataFrame with encoded categoricals.
    """
    df = df.copy()

    # Education — ordinal encoding
    if "Education" in df.columns:
        df["Education_Level"] = df["Education"].map(EDUCATION_ORDER)
        if df["Education_Level"].isna().any():
            unknown = df.loc[df["Education_Level"].isna(), "Education"].unique()
            logger.warning(f"Unknown education values encountered: {unknown}. Defaulting to 3.")
            df["Education_Level"] = df["Education_Level"].fillna(3)
        df["Education_Level"] = df["Education_Level"].astype(int)
        df = df.drop(columns=["Education"])

    # Gender — binary encoding
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(GENDER_MAP)
        if df["Gender"].isna().any():
            logger.warning("Unknown gender values. Defaulting to 0.")
            df["Gender"] = df["Gender"].fillna(0).astype(int)

    # Marital Status — binary encoding
    if "Marital Status" in df.columns:
        df["Marital_Status"] = df["Marital Status"].map(MARITAL_STATUS_MAP)
        df = df.drop(columns=["Marital Status"])
    elif "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].map(MARITAL_STATUS_MAP)

    # Home Ownership — binary encoding
    if "Home Ownership" in df.columns:
        df["Home_Ownership"] = df["Home Ownership"].map(HOME_OWNERSHIP_MAP)
        df = df.drop(columns=["Home Ownership"])
    elif "Home_Ownership" in df.columns:
        df["Home_Ownership"] = df["Home_Ownership"].map(HOME_OWNERSHIP_MAP)

    # Number of Children — rename for consistency
    if "Number of Children" in df.columns:
        df = df.rename(columns={"Number of Children": "Number_of_Children"})

    logger.debug("Categorical encoding complete.")
    return df


def reorder_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to match the exact feature order used during training.

    Args:
        df: DataFrame with all processed features.

    Returns:
        DataFrame with columns in FINAL_FEATURE_ORDER.

    Raises:
        KeyError: If required features are missing.
    """
    missing = set(FINAL_FEATURE_ORDER) - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required features after preprocessing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[FINAL_FEATURE_ORDER]


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply StandardScaler to numerical features using the saved scaler.

    The scaler was fitted during Colab training and saved to models/scaler.pkl.
    Only numerical features are scaled; binary/ordinal features are left intact.

    Args:
        df: DataFrame with all features in final order.

    Returns:
        DataFrame with numerical features scaled.

    Raises:
        FileNotFoundError: If scaler.pkl is not found.
    """
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        logger.error(
            f"Scaler not found at {SCALER_PATH}. "
            "Please train the model on Colab and place scaler.pkl in models/."
        )
        raise

    # Scale only numerical columns
    numerical_cols_present = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df = df.copy()
    df[numerical_cols_present] = scaler.transform(df[numerical_cols_present])

    logger.debug(f"Scaled {len(numerical_cols_present)} numerical features.")
    return df


def preprocess(input_data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Full preprocessing pipeline — from raw input to model-ready features.

    This is the single entry point used by both the API and dashboard.
    It chains: engineer → encode → reorder → scale.

    Args:
        input_data: Either a dict (single prediction) or DataFrame (batch).

    Returns:
        Preprocessed DataFrame ready for model.predict().
    """
    # Convert dict to single-row DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    logger.info(f"Preprocessing {len(df)} sample(s)...")

    # Pipeline
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = reorder_features(df)
    df = scale_features(df)

    logger.info("Preprocessing complete.")
    return df
