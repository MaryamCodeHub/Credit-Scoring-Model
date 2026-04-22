"""
Prediction Engine — Credit Scoring System.

Loads the trained model and artifacts from the models/ directory,
then provides a clean interface for making predictions.

Usage:
    from src.predict import CreditScorer
    scorer = CreditScorer()
    result = scorer.predict({
        "Age": 30, "Gender": "Male", "Income": 75000,
        "Education": "Bachelor's Degree", "Marital Status": "Single",
        "Number of Children": 0, "Home Ownership": "Rented"
    })
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Union

from src.config import MODEL_PATH, TARGET_ENCODER_PATH, FEATURE_CONFIG_PATH
from src.preprocessing import preprocess
from src.logger import logger


class CreditScorer:
    """
    Production credit scoring inference engine.

    Loads model artifacts once on initialization, then provides
    fast predictions via the predict() method.
    """

    def __init__(self):
        """Load all model artifacts from disk."""
        self.model = None
        self.target_encoder = None
        self.feature_config = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load model, target encoder, and feature config from models/ directory.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        try:
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")

            self.target_encoder = joblib.load(TARGET_ENCODER_PATH)
            logger.info(f"✅ Target encoder loaded from {TARGET_ENCODER_PATH}")

            if FEATURE_CONFIG_PATH.exists():
                with open(FEATURE_CONFIG_PATH, "r") as f:
                    self.feature_config = json.load(f)
                logger.info(f"✅ Feature config loaded from {FEATURE_CONFIG_PATH}")
            else:
                logger.warning(
                    f"Feature config not found at {FEATURE_CONFIG_PATH}. "
                    "Using default feature order from config.py."
                )

        except FileNotFoundError as e:
            logger.error(
                f"❌ Missing model artifact: {e}. "
                "Please train the model on Colab and place artifacts in models/."
            )
            raise

    def predict(self, input_data: Union[dict, pd.DataFrame]) -> dict:
        """
        Make a credit score prediction.

        Args:
            input_data: Raw applicant data (dict for single, DataFrame for batch).

        Returns:
            dict with keys:
                - credit_score: str ("Low", "Average", or "High")
                - confidence: float (max probability)
                - probabilities: dict mapping each class to its probability
                - risk_level: str ("High Risk", "Medium Risk", or "Low Risk")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_artifacts() first.")

        # Preprocess
        features = preprocess(input_data)
        logger.debug(f"Feature vector shape: {features.shape}")

        # Predict class
        prediction = self.model.predict(features)
        predicted_label = self.target_encoder.inverse_transform(prediction)[0]

        # Predict probabilities
        probabilities = self.model.predict_proba(features)[0]
        class_names = self.target_encoder.classes_
        prob_dict = {
            class_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(class_names))
        }

        # Confidence = max probability
        confidence = round(float(np.max(probabilities)), 4)

        # Risk level mapping
        risk_map = {"Low": "High Risk", "Average": "Medium Risk", "High": "Low Risk"}
        risk_level = risk_map.get(predicted_label, "Unknown")

        result = {
            "credit_score": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "risk_level": risk_level,
        }

        logger.info(
            f"Prediction: {predicted_label} (confidence: {confidence:.2%}) — {risk_level}"
        )

        return result

    def predict_batch(self, data: pd.DataFrame) -> list[dict]:
        """
        Make predictions for multiple applicants.

        Args:
            data: DataFrame with multiple rows of applicant data.

        Returns:
            List of prediction dicts (same format as predict()).
        """
        results = []
        for idx in range(len(data)):
            row = data.iloc[[idx]]
            result = self.predict(row)
            results.append(result)

        logger.info(f"Batch prediction complete: {len(results)} samples processed.")
        return results

    @property
    def is_loaded(self) -> bool:
        """Check if model artifacts are loaded and ready."""
        return self.model is not None and self.target_encoder is not None
