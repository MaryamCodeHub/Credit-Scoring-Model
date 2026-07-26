"""Integrity-checked inference service for the frozen credit model v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_V2_METADATA_PATH, MODEL_V2_PATH
from src.model_preprocessing import (
    INPUT_NUMERICAL_FEATURES,
    MODEL_FEATURES,
    CreditModelPreprocessor,
)


EXPECTED_MODEL_VERSION = "2.0.0"
EXPECTED_TRANSFORMED_FEATURE_COUNT = 35
EXPECTED_CLASSES = frozenset({"Good", "Poor", "Standard"})
EXPECTED_CLASSIFIER_PARAMETERS = {
    "max_depth": 6,
    "min_samples_leaf": 100,
    "class_weight": None,
    "random_state": 42,
}
RISK_BY_CLASS = {
    "Poor": "High Risk",
    "Standard": "Medium Risk",
    "Good": "Low Risk",
}


class ModelV2Error(RuntimeError):
    """Base error for safe v2 model loading and inference."""


class ModelV2UnavailableError(ModelV2Error):
    """Raised when a required artifact is unavailable."""


class ModelV2IntegrityError(ModelV2Error):
    """Raised when the model artifact fails its recorded hash check."""


class ModelV2CompatibilityError(ModelV2Error):
    """Raised when artifact structure or metadata violates the contract."""


class ModelV2InferenceError(ModelV2Error):
    """Raised when a loaded artifact returns an invalid prediction."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelV2UnavailableError("Model v2 metadata is unavailable.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelV2CompatibilityError("Model v2 metadata is invalid.") from error
    if not isinstance(metadata, dict):
        raise ModelV2CompatibilityError("Model v2 metadata must be an object.")
    return metadata


def _require_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ModelV2CompatibilityError("Unexpected model v2 version.")
    if metadata.get("model_type") != "DecisionTreeClassifier":
        raise ModelV2CompatibilityError("Unexpected classifier type.")
    if metadata.get("transformed_feature_count") != EXPECTED_TRANSFORMED_FEATURE_COUNT:
        raise ModelV2CompatibilityError("Unexpected transformed feature count.")
    if metadata.get("raw_model_features") != list(MODEL_FEATURES):
        raise ModelV2CompatibilityError("Unexpected raw model feature contract.")
    if set(metadata.get("target_classes", [])) != EXPECTED_CLASSES:
        raise ModelV2CompatibilityError("Unexpected model class set.")
    recorded_hash = metadata.get("model_artifact_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ModelV2CompatibilityError("Model artifact hash is missing or invalid.")

    parameters = metadata.get("model_parameters")
    if not isinstance(parameters, Mapping):
        raise ModelV2CompatibilityError("Classifier parameters are missing.")
    for name, expected in EXPECTED_CLASSIFIER_PARAMETERS.items():
        if parameters.get(name) != expected:
            raise ModelV2CompatibilityError(
                f"Unexpected frozen classifier parameter: {name}."
            )


def _iter_estimators(root: BaseEstimator) -> list[BaseEstimator]:
    discovered: list[BaseEstimator] = []
    seen: set[int] = set()

    def visit(estimator: BaseEstimator) -> None:
        if id(estimator) in seen:
            return
        seen.add(id(estimator))
        discovered.append(estimator)
        for value in estimator.get_params(deep=True).values():
            if isinstance(value, BaseEstimator):
                visit(value)

    visit(root)
    return discovered


def _validate_pipeline(pipeline: object) -> Pipeline:
    if not isinstance(pipeline, Pipeline):
        raise ModelV2CompatibilityError("Model artifact is not a sklearn Pipeline.")
    if list(pipeline.named_steps) != ["preprocessing", "classifier"]:
        raise ModelV2CompatibilityError("Unexpected model pipeline steps.")

    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessing, CreditModelPreprocessor):
        raise ModelV2CompatibilityError("Unexpected preprocessing component.")
    if not isinstance(classifier, DecisionTreeClassifier):
        raise ModelV2CompatibilityError("Unexpected classifier component.")

    parameters = classifier.get_params()
    for name, expected in EXPECTED_CLASSIFIER_PARAMETERS.items():
        if parameters.get(name) != expected:
            raise ModelV2CompatibilityError(
                f"Loaded classifier parameter does not match: {name}."
            )

    try:
        transformed_names = preprocessing.get_feature_names_out()
    except Exception as error:
        raise ModelV2CompatibilityError(
            "Fitted preprocessing feature names are unavailable."
        ) from error
    if len(transformed_names) != EXPECTED_TRANSFORMED_FEATURE_COUNT:
        raise ModelV2CompatibilityError("Loaded transformed feature count is invalid.")

    classes = getattr(classifier, "classes_", None)
    if classes is None or set(map(str, classes)) != EXPECTED_CLASSES:
        raise ModelV2CompatibilityError("Loaded classifier classes are invalid.")

    estimators = _iter_estimators(pipeline)
    if any(isinstance(estimator, BaseSearchCV) for estimator in estimators):
        raise ModelV2CompatibilityError("Search objects are not permitted.")
    decision_trees = [
        estimator
        for estimator in estimators
        if isinstance(estimator, DecisionTreeClassifier)
    ]
    classifiers = [
        estimator
        for estimator in estimators
        if estimator is not pipeline and is_classifier(estimator)
    ]
    if decision_trees != [classifier] or classifiers != [classifier]:
        raise ModelV2CompatibilityError(
            "The artifact must contain exactly one approved classifier."
        )
    return pipeline


class CreditScoreV2Service:
    """Load once after verification and serve predictions without fitting."""

    def __init__(
        self,
        model_path: Path = MODEL_V2_PATH,
        metadata_path: Path = MODEL_V2_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = _load_metadata(self.metadata_path)
        _require_metadata(self.metadata)

        if not self.model_path.is_file():
            raise ModelV2UnavailableError("Model v2 artifact is unavailable.")
        self.artifact_sha256 = _sha256(self.model_path)
        if self.artifact_sha256 != self.metadata["model_artifact_sha256"]:
            raise ModelV2IntegrityError("Model v2 artifact hash verification failed.")

        try:
            loaded = joblib.load(self.model_path)
        except Exception as error:
            raise ModelV2CompatibilityError(
                "Model v2 artifact could not be loaded."
            ) from error
        self.pipeline = _validate_pipeline(loaded)
        self.classifier = self.pipeline.named_steps["classifier"]
        self.model_version = EXPECTED_MODEL_VERSION

    def predict(self, features: Mapping[str, object]) -> dict[str, object]:
        """Predict one schema-validated profile with the already-fitted pipeline."""
        if set(features) != set(MODEL_FEATURES):
            missing = sorted(set(MODEL_FEATURES) - set(features))
            extra = sorted(set(features) - set(MODEL_FEATURES))
            raise ModelV2InferenceError(
                f"Invalid model feature contract; missing={missing}, extra={extra}."
            )

        row = {
            feature: np.nan if features[feature] is None else features[feature]
            for feature in MODEL_FEATURES
        }
        frame = pd.DataFrame([row], columns=MODEL_FEATURES)
        try:
            for feature in INPUT_NUMERICAL_FEATURES:
                frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(
                    float
                )
        except (TypeError, ValueError) as error:
            raise ModelV2InferenceError("Numerical model input is invalid.") from error

        # The API deliberately does not duplicate the six audited extreme rules.
        # Finite, type-valid values flow into the frozen pipeline, whose fitted
        # ExtremeValueTransformer performs the approved masking consistently.
        try:
            predicted_values = self.pipeline.predict(frame)
            probability_rows = self.pipeline.predict_proba(frame)
        except Exception as error:
            raise ModelV2InferenceError("Model v2 inference failed.") from error

        if len(predicted_values) != 1 or np.asarray(probability_rows).shape != (1, 3):
            raise ModelV2InferenceError("Model v2 returned an unexpected output shape.")
        predicted = str(predicted_values[0])
        class_names = [str(label) for label in self.classifier.classes_]
        values = np.asarray(probability_rows[0], dtype=float)
        if (
            predicted not in EXPECTED_CLASSES
            or set(class_names) != EXPECTED_CLASSES
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.isclose(values.sum(), 1.0, rtol=1e-6, atol=1e-8)
        ):
            raise ModelV2InferenceError("Model v2 returned invalid probabilities.")

        by_class = dict(zip(class_names, values.tolist(), strict=True))
        confidence = float(by_class[predicted])
        return {
            "credit_score": predicted,
            "probabilities": {
                "poor": float(by_class["Poor"]),
                "standard": float(by_class["Standard"]),
                "good": float(by_class["Good"]),
            },
            "confidence": confidence,
            "risk_level": RISK_BY_CLASS[predicted],
            "model_version": self.model_version,
        }
