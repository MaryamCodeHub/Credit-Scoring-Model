"""Pure dashboard data contracts, integrity checks, and Plotly components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import plotly.graph_objects as go

from dashboard.api_client import ApiStatus, PredictionResult
from dashboard.theme import (
    CLASS_COLORS,
    POOR,
    PRIMARY_TEAL,
    SECONDARY_BLUE,
    STANDARD,
    plotly_layout,
)


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
ANALYTICS_PATH = DASHBOARD_DIR / "data" / "development_analytics_v1.json"
ANALYTICS_SHA256_PATH = (
    DASHBOARD_DIR / "data" / "development_analytics_v1.sha256"
)
MODEL_METADATA_PATH = (
    PROJECT_ROOT / "models" / "v2" / "model_metadata.json"
)

NAVIGATION_SECTIONS = (
    "Executive Overview",
    "Credit Prediction",
    "Data Insights",
    "Model Performance",
    "Model Card & Limitations",
)

PREDICTION_FIELDS = (
    "age",
    "annual_income",
    "monthly_inhand_salary",
    "num_bank_accounts",
    "num_credit_cards",
    "num_loans",
    "num_delayed_payments",
    "changed_credit_limit",
    "num_credit_inquiries",
    "outstanding_debt",
    "credit_utilization_ratio",
    "credit_history_age_months",
    "total_emi_per_month",
    "occupation",
)

OPTIONAL_PREDICTION_FIELDS = (
    "monthly_inhand_salary",
    "changed_credit_limit",
    "num_credit_inquiries",
    "occupation",
)

EDUCATIONAL_DISCLAIMER = (
    "Educational portfolio project only. This dashboard is not approved for "
    "real lending decisions and does not replace qualified human review."
)

CLASS_ORDER = ("Poor", "Standard", "Good")
VALIDATION_MACRO_F1 = 0.619834
FINAL_METRICS = {
    "macro_f1": 0.6288978544,
    "accuracy": 0.6742021277,
    "balanced_accuracy": 0.6149644096,
    "weighted_f1": 0.6688847035,
    "macro_precision": 0.6533535038,
    "macro_recall": 0.6149644096,
    "poor_recall": 0.5757722008,
}
PARTITION_SIZES = {
    "labeled_dataset": {"rows": 100_000, "customers": 12_500},
    "development": {"rows": 71_424, "customers": 8_928},
    "validation": {"rows": 14_288, "customers": 1_786},
    "final_test": {"rows": 14_288, "customers": 1_786},
}
PER_CLASS_METRICS = {
    "Poor": {
        "precision": 0.735965,
        "recall": 0.575772,
        "f1": 0.646087,
        "support": 4_144,
    },
    "Standard": {
        "precision": 0.686250,
        "recall": 0.795024,
        "f1": 0.736643,
        "support": 7_596,
    },
    "Good": {
        "precision": 0.537845,
        "recall": 0.474097,
        "f1": 0.503963,
        "support": 2_548,
    },
}
CONFUSION_MATRIX = (
    (2_386, 1_445, 313),
    (832, 6_039, 725),
    (24, 1_316, 1_208),
)


class AnalyticsIntegrityError(RuntimeError):
    """Raised when aggregate analytics cannot be verified safely."""


class MetadataIntegrityError(RuntimeError):
    """Raised when committed model metadata violates the dashboard contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_verified_analytics(
    json_path: Path = ANALYTICS_PATH,
    sidecar_path: Path = ANALYTICS_SHA256_PATH,
) -> dict[str, Any]:
    """Verify exact JSON bytes before parsing any aggregate analytics."""
    try:
        expected = sidecar_path.read_text(encoding="ascii").strip().lower()
        payload = json_path.read_bytes()
    except OSError:
        raise AnalyticsIntegrityError(
            "Verified development analytics are unavailable."
        ) from None
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise AnalyticsIntegrityError("The analytics checksum is invalid.")
    if _sha256_bytes(payload) != expected:
        raise AnalyticsIntegrityError(
            "The analytics integrity check failed. Insights are disabled."
        )
    try:
        analytics = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise AnalyticsIntegrityError("The analytics artifact is malformed.") from None
    if (
        not isinstance(analytics, dict)
        or analytics.get("source_partition") != "development_train"
        or analytics.get("privacy", {}).get("aggregate_only") is not True
        or analytics.get("privacy", {}).get("suppression_threshold") != 25
    ):
        raise AnalyticsIntegrityError(
            "The analytics privacy contract is invalid."
        )
    return analytics


def load_model_metadata(
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
    """Load committed, aggregate model evidence without loading the model."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MetadataIntegrityError("Model metadata are unavailable.") from None
    if (
        metadata.get("model_version") != "2.0.0"
        or metadata.get("model_type") != "DecisionTreeClassifier"
        or metadata.get("transformed_feature_count") != 35
        or metadata.get("final_test_evaluated_once") is not True
        or metadata.get("portfolio_acceptance") is not True
    ):
        raise MetadataIntegrityError("Model metadata do not match version 2.")
    return metadata


def build_prediction_payload(values: Mapping[str, object]) -> dict[str, object]:
    """Return the exact public v2 payload, preserving optional JSON nulls."""
    missing = sorted(set(PREDICTION_FIELDS) - set(values))
    extra = sorted(set(values) - set(PREDICTION_FIELDS))
    if missing or extra:
        raise ValueError(
            f"Prediction fields do not match the v2 contract; "
            f"missing={missing}, extra={extra}."
        )
    return {field: values[field] for field in PREDICTION_FIELDS}


def prediction_is_available(status: ApiStatus) -> bool:
    return bool(status.available and status.ready)


def probability_figure(result: PredictionResult) -> go.Figure:
    probabilities = result.probabilities or {}
    labels = list(CLASS_ORDER)
    values = [100 * probabilities.get(label.lower(), 0.0) for label in labels]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[CLASS_COLORS[label] for label in labels],
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Predicted class probabilities",
            x_title="Probability (%)",
            y_title="Credit-score category",
            height=300,
        )
    )
    figure.update_xaxes(range=[0, 100])
    return figure


def target_distribution_figure(analytics: Mapping[str, Any]) -> go.Figure:
    distribution = analytics["target_distribution"]
    percentages = [distribution[label]["percentage"] for label in CLASS_ORDER]
    figure = go.Figure(
        go.Bar(
            x=list(CLASS_ORDER),
            y=percentages,
            marker_color=[CLASS_COLORS[label] for label in CLASS_ORDER],
            text=[f"{value:.1f}%" for value in percentages],
            textposition="outside",
            customdata=[distribution[label]["count"] for label in CLASS_ORDER],
            hovertemplate=(
                "%{x}<br>Share: %{y:.2f}%<br>Records: %{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Development target distribution",
            x_title="Credit-score category",
            y_title="Development records (%)",
        )
    )
    return figure


def _bin_labels(edges: list[float]) -> list[str]:
    return [
        f"{edges[index]:,.0f}–{edges[index + 1]:,.0f}"
        for index in range(len(edges) - 1)
    ]


def histogram_figure(
    analytics: Mapping[str, Any],
    feature: str,
    *,
    title: str,
    x_title: str,
) -> go.Figure:
    histogram = analytics["histograms"][feature]
    labels = _bin_labels(histogram["bin_edges"])
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=histogram["percentages"],
            marker_color=SECONDARY_BLUE,
            customdata=histogram["counts"],
            hovertemplate=(
                "%{x}<br>Share: %{y:.2f}%<br>Records: %{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **plotly_layout(
            title=title,
            x_title=x_title,
            y_title="Development records (%)",
        )
    )
    return figure


def occupation_figure(analytics: Mapping[str, Any]) -> go.Figure:
    distribution = analytics["occupation_distribution"]
    items = sorted(
        distribution.items(), key=lambda item: item[1]["percentage"]
    )
    figure = go.Figure(
        go.Bar(
            x=[item[1]["percentage"] for item in items],
            y=[item[0].replace("_", " ") for item in items],
            orientation="h",
            marker_color=PRIMARY_TEAL,
            customdata=[item[1]["count"] for item in items],
            hovertemplate=(
                "%{y}<br>Share: %{x:.2f}%<br>Records: %{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Occupation distribution",
            x_title="Development records (%)",
            y_title="Occupation",
            height=510,
        )
    )
    return figure


def missing_values_figure(analytics: Mapping[str, Any]) -> go.Figure:
    missing = sorted(
        analytics["missing_values"].items(),
        key=lambda item: item[1]["percentage"],
    )
    figure = go.Figure(
        go.Bar(
            x=[item[1]["percentage"] for item in missing],
            y=[item[0] for item in missing],
            orientation="h",
            marker_color=STANDARD,
            customdata=[item[1]["count"] for item in missing],
            hovertemplate=(
                "%{y}<br>Missing: %{x:.2f}%<br>Records: %{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Missing values in approved model inputs",
            x_title="Missing values (%)",
            y_title="Approved input feature",
            height=470,
        )
    )
    return figure


def extreme_invalid_figure(analytics: Mapping[str, Any]) -> go.Figure:
    invalid = analytics["extreme_invalid_counts"]
    features = list(invalid)
    figure = go.Figure(
        go.Bar(
            x=features,
            y=[invalid[feature]["percentage"] for feature in features],
            marker_color=POOR,
            customdata=[
                [invalid[feature]["count"], invalid[feature]["threshold"]]
                for feature in features
            ],
            hovertemplate=(
                "%{x}<br>Flagged: %{y:.2f}%<br>Records: %{customdata[0]:,}"
                "<br>Audited threshold: %{customdata[1]:,.2f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Audited extreme-invalid values",
            x_title="Approved audited rule",
            y_title="Development records flagged (%)",
            height=420,
        )
    )
    return figure


def correlation_figure(analytics: Mapping[str, Any]) -> go.Figure:
    correlation = analytics["correlation_matrix"]
    figure = go.Figure(
        go.Heatmap(
            z=correlation["values"],
            x=correlation["features"],
            y=correlation["features"],
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, SECONDARY_BLUE],
                [0.5, "#FFFFFF"],
                [1.0, POOR],
            ],
            colorbar={"title": "Pearson r"},
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Numerical correlation matrix",
            x_title="Approved numerical feature",
            y_title="Approved numerical feature",
            height=650,
        )
    )
    return figure


def bivariate_figure(table: Mapping[str, Any], *, title: str) -> go.Figure:
    x_edges = table["x_bin_edges"]
    y_edges = table["y_bin_edges"]
    matrix: list[list[float | None]] = [
        [None] * (len(x_edges) - 1) for _ in range(len(y_edges) - 1)
    ]
    for cell in table["published_cells"]:
        matrix[cell["y_bin_index"]][cell["x_bin_index"]] = cell["percentage"]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=_bin_labels(x_edges),
            y=_bin_labels(y_edges),
            colorscale=[[0.0, "#CCFBF1"], [1.0, PRIMARY_TEAL]],
            colorbar={"title": "Records (%)"},
            hovertemplate=(
                "X bin: %{x}<br>Y bin: %{y}<br>Share: %{z:.2f}%<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        **plotly_layout(
            title=title,
            x_title=table["x_feature"],
            y_title=table["y_feature"],
            height=470,
        )
    )
    return figure


def per_class_metrics_figure() -> go.Figure:
    figure = go.Figure()
    colors = [SECONDARY_BLUE, PRIMARY_TEAL, STANDARD]
    for metric, color in zip(("precision", "recall", "f1"), colors, strict=True):
        figure.add_trace(
            go.Bar(
                name=metric.title(),
                x=list(CLASS_ORDER),
                y=[PER_CLASS_METRICS[label][metric] for label in CLASS_ORDER],
                marker_color=color,
                text=[
                    f"{PER_CLASS_METRICS[label][metric]:.3f}"
                    for label in CLASS_ORDER
                ],
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.4f}<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="group",
        **plotly_layout(
            title="Per-class evaluation metrics",
            x_title="Credit-score category",
            y_title="Metric value",
        ),
    )
    figure.update_yaxes(range=[0, 1])
    return figure


def confusion_matrix_figure() -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=CONFUSION_MATRIX,
            x=list(CLASS_ORDER),
            y=list(CLASS_ORDER),
            text=CONFUSION_MATRIX,
            texttemplate="%{text:,}",
            colorscale=[[0.0, "#E2E8F0"], [1.0, PRIMARY_TEAL]],
            colorbar={"title": "Predictions"},
            hovertemplate=(
                "Actual: %{y}<br>Predicted: %{x}<br>Records: %{z:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Final-test confusion matrix",
            x_title="Predicted class",
            y_title="Actual class",
            height=440,
        )
    )
    return figure


def validation_final_figure() -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=["Validation", "Final test"],
            y=[VALIDATION_MACRO_F1, FINAL_METRICS["macro_f1"]],
            marker_color=[SECONDARY_BLUE, PRIMARY_TEAL],
            text=[
                f"{VALIDATION_MACRO_F1:.4f}",
                f"{FINAL_METRICS['macro_f1']:.4f}",
            ],
            textposition="outside",
            hovertemplate="%{x}<br>Macro F1: %{y:.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        **plotly_layout(
            title="Validation versus final-test Macro F1",
            x_title="Evaluation partition",
            y_title="Macro F1",
            height=360,
        )
    )
    figure.update_yaxes(range=[0, 1])
    return figure
