"""Build a deterministic, privacy-safe development analytics artifact.

This script verifies the labeled Kaggle training file before parsing it,
recreates the customer-grouped split, and computes aggregate analytics from
``development_train`` only. It never writes row-level or transformed data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import RANDOM_STATE, ROOT_DIR
from src.data_cleaning import clean_credit_data
from src.data_splitting import split_by_customer
from src.extreme_values import EXTREME_VALUE_THRESHOLDS
from src.model_preprocessing import (
    GROUP_COLUMN,
    INPUT_NUMERICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
)


EXPECTED_TRAIN_SHA256: Final = (
    "D2EBCC056A64C48710B1AEB96777155835D372D7AD202529F64666011D214DA0"
)
DEFAULT_SOURCE_PATH: Final = (
    ROOT_DIR / "data" / "raw" / "kaggle_credit_score" / "train.csv"
)
DEFAULT_OUTPUT_PATH: Final = (
    ROOT_DIR / "dashboard" / "data" / "development_analytics_v1.json"
)
DEFAULT_SIDECAR_PATH: Final = (
    ROOT_DIR / "dashboard" / "data" / "development_analytics_v1.sha256"
)
SCHEMA_VERSION: Final = "1.0.0"
SOURCE_PARTITION: Final = "development_train"
SUPPRESSION_THRESHOLD: Final = 25
TARGET_CLASSES: Final = ("Poor", "Standard", "Good")
CORRELATION_METHOD: Final = "pearson_pairwise_complete"
ROUND_DIGITS: Final = 6
GENERATOR_COMMIT_PATHS: Final = (
    "scripts/build_dashboard_analytics.py",
    "tests/test_analytics_artifact.py",
)

HISTOGRAM_EDGES: Final = {
    "Age": (0.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 121.0),
    "Annual_Income": (
        0.0,
        25_000.0,
        50_000.0,
        75_000.0,
        100_000.0,
        150_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
        5_000_000.0,
        25_000_000.0,
    ),
    "Outstanding_Debt": (
        0.0,
        500.0,
        1_000.0,
        1_500.0,
        2_000.0,
        2_500.0,
        3_000.0,
        4_000.0,
        5_000.0,
    ),
    "Credit_Utilization_Ratio": (
        0.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
    ),
}

BIVARIATE_DEFINITIONS: Final = (
    (
        "Annual_Income_vs_Outstanding_Debt",
        "Annual_Income",
        "Outstanding_Debt",
        HISTOGRAM_EDGES["Annual_Income"],
        HISTOGRAM_EDGES["Outstanding_Debt"],
    ),
    (
        "Credit_Utilization_Ratio_vs_Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Outstanding_Debt",
        HISTOGRAM_EDGES["Credit_Utilization_Ratio"],
        HISTOGRAM_EDGES["Outstanding_Debt"],
    ),
)


def calculate_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _rounded(value: float) -> float:
    rounded = round(float(value), ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _percentage(count: int, total: int) -> float:
    return _rounded(100.0 * count / total) if total else 0.0


def _normalize_timestamp(value: str | None) -> str:
    if value is None:
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if source_date_epoch is not None:
            parsed = datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc)
        else:
            parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("generated_at_utc must include a timezone.")
        parsed = parsed.astimezone(timezone.utc)
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_generator_commit(value: str) -> str:
    """Validate a frozen local generator commit without exposing Git details."""
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(
            "generator commit must be a full 40-character hexadecimal hash."
        )
    commit = value.lower()
    try:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        if exists.returncode != 0:
            raise ValueError("generator commit does not exist locally.")

        tree = subprocess.run(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                commit,
                "--",
                *GENERATOR_COMMIT_PATHS,
            ],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError("generator commit could not be validated locally.") from None

    present_paths = set(tree.stdout.splitlines()) if tree.returncode == 0 else set()
    if set(GENERATOR_COMMIT_PATHS) - present_paths:
        raise ValueError(
            "generator commit does not contain the required analytics files."
        )
    return commit


def _target_distribution(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    counts = frame[TARGET_COLUMN].value_counts()
    unexpected = sorted(set(counts.index) - set(TARGET_CLASSES))
    if unexpected:
        raise ValueError(f"Unexpected target classes: {unexpected}")
    total = len(frame)
    return {
        target_class: {
            "count": int(counts.get(target_class, 0)),
            "percentage": _percentage(int(counts.get(target_class, 0)), total),
        }
        for target_class in TARGET_CLASSES
    }


def _summary_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for feature in INPUT_NUMERICAL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = values.dropna()
        result[feature] = {
            "count": int(valid.size),
            "missing_count": int(values.isna().sum()),
            "minimum": _rounded(valid.min()) if not valid.empty else None,
            "median": _rounded(valid.median()) if not valid.empty else None,
            "mean": _rounded(valid.mean()) if not valid.empty else None,
            "maximum": _rounded(valid.max()) if not valid.empty else None,
        }
    return result


def _fixed_histogram(
    values: pd.Series,
    edges: tuple[float, ...],
    *,
    row_count: int,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().to_numpy(dtype=float)
    counts, returned_edges = np.histogram(valid, bins=np.asarray(edges, dtype=float))
    if int(counts.sum()) != len(valid):
        raise ValueError("Fixed histogram edges do not cover every non-missing value.")
    count_values = [int(value) for value in counts]
    return {
        "bin_edges": [float(value) for value in returned_edges],
        "counts": count_values,
        "percentages": [_percentage(value, row_count) for value in count_values],
        "missing_count": int(numeric.isna().sum()),
    }


def _occupation_distribution(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    values = frame["Occupation"].fillna("Unknown").astype(str)
    counts = values.value_counts()
    rare = counts[counts < SUPPRESSION_THRESHOLD]
    published = counts[counts >= SUPPRESSION_THRESHOLD].to_dict()
    if not rare.empty:
        published["Other"] = int(rare.sum())
    return {
        category: {
            "count": int(count),
            "percentage": _percentage(int(count), len(frame)),
        }
        for category, count in sorted(published.items())
    }


def _missing_values(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {
        feature: {
            "count": int(frame[feature].isna().sum()),
            "percentage": _percentage(int(frame[feature].isna().sum()), len(frame)),
        }
        for feature in MODEL_FEATURES
    }


def _extreme_invalid_counts(
    frame: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for feature, threshold in EXTREME_VALUE_THRESHOLDS.items():
        values = pd.to_numeric(frame[feature], errors="coerce")
        count = int((values.notna() & values.gt(threshold)).sum())
        result[feature] = {
            "threshold": float(threshold),
            "count": count,
            "percentage": _percentage(count, len(frame)),
        }
    return result


def _correlation_matrix(frame: pd.DataFrame) -> dict[str, Any]:
    features = list(INPUT_NUMERICAL_FEATURES)
    numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.corr(method="pearson")
    values: list[list[float | None]] = []
    for row in matrix.to_numpy(dtype=float):
        values.append(
            [None if not np.isfinite(value) else _rounded(value) for value in row]
        )
    return {
        "method": CORRELATION_METHOD,
        "features": features,
        "values": values,
    }


def _bivariate_bins(
    frame: pd.DataFrame,
    *,
    name: str,
    x_feature: str,
    y_feature: str,
    x_edges: tuple[float, ...],
    y_edges: tuple[float, ...],
) -> tuple[dict[str, Any], int]:
    pairs = frame.loc[:, [x_feature, y_feature]].apply(
        pd.to_numeric, errors="coerce"
    )
    complete = pairs.dropna()
    matrix, returned_x_edges, returned_y_edges = np.histogram2d(
        complete[x_feature].to_numpy(dtype=float),
        complete[y_feature].to_numpy(dtype=float),
        bins=[np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)],
    )
    integer_matrix = matrix.astype(np.int64)
    if int(integer_matrix.sum()) != len(complete):
        raise ValueError(f"Fixed bivariate edges do not cover all values for {name}.")

    published_cells: list[dict[str, float | int]] = []
    suppressed_cell_count = 0
    suppressed_observation_count = 0
    for x_index in range(integer_matrix.shape[0]):
        for y_index in range(integer_matrix.shape[1]):
            count = int(integer_matrix[x_index, y_index])
            if count < SUPPRESSION_THRESHOLD:
                suppressed_cell_count += 1
                suppressed_observation_count += count
                continue
            published_cells.append(
                {
                    "x_bin_index": x_index,
                    "y_bin_index": y_index,
                    "count": count,
                    "percentage": _percentage(count, len(frame)),
                }
            )

    return (
        {
            "x_feature": x_feature,
            "y_feature": y_feature,
            "x_bin_edges": [float(value) for value in returned_x_edges],
            "y_bin_edges": [float(value) for value in returned_y_edges],
            "published_cells": published_cells,
            "suppressed_cell_count": suppressed_cell_count,
            "suppressed_observation_count": suppressed_observation_count,
            "missing_pair_count": int(len(frame) - len(complete)),
        },
        suppressed_cell_count,
    )


def build_analytics(
    labeled_data: pd.DataFrame,
    *,
    generated_at_utc: str,
    generator_commit: str,
    source_dataset_sha256: str = EXPECTED_TRAIN_SHA256,
) -> dict[str, Any]:
    """Build aggregate analytics while touching only the development partition."""
    partitions = split_by_customer(labeled_data, random_state=RANDOM_STATE)
    development = clean_credit_data(partitions.development_train)
    row_count = len(development)
    group_sizes = development.groupby(GROUP_COLUMN, sort=False).size()

    histograms = {
        feature: _fixed_histogram(
            development[feature],
            edges,
            row_count=row_count,
        )
        for feature, edges in HISTOGRAM_EDGES.items()
    }

    bivariate: dict[str, Any] = {}
    total_suppressed = 0
    for name, x_feature, y_feature, x_edges, y_edges in BIVARIATE_DEFINITIONS:
        result, suppressed_count = _bivariate_bins(
            development,
            name=name,
            x_feature=x_feature,
            y_feature=y_feature,
            x_edges=x_edges,
            y_edges=y_edges,
        )
        bivariate[name] = result
        total_suppressed += suppressed_count

    analytics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "generator_commit": generator_commit,
        "source_partition": SOURCE_PARTITION,
        "split_seed": RANDOM_STATE,
        "source_dataset_sha256": source_dataset_sha256,
        "row_count": row_count,
        "customer_count": int(group_sizes.size),
        "minimum_group_size": int(group_sizes.min()),
        "privacy": {
            "aggregate_only": True,
            "suppression_threshold": SUPPRESSION_THRESHOLD,
            "forbidden_fields_absent": True,
        },
        "target_distribution": _target_distribution(development),
        "summary_statistics": _summary_statistics(development),
        "histograms": histograms,
        "occupation_distribution": _occupation_distribution(development),
        "missing_values": _missing_values(development),
        "extreme_invalid_counts": _extreme_invalid_counts(development),
        "correlation_matrix": _correlation_matrix(development),
        "bivariate_bins": bivariate,
        "suppressed_bin_count": total_suppressed,
    }
    _validate_privacy(analytics)
    return analytics


def _validate_privacy(analytics: dict[str, Any]) -> None:
    serialized = json.dumps(analytics, sort_keys=True, allow_nan=False)
    forbidden = ('"ID"', '"Customer_ID"', '"Name"', '"SSN"')
    if any(value in serialized for value in forbidden):
        raise ValueError("A forbidden identifier or PII field entered the artifact.")
    if any(marker in serialized for marker in (":\\", "/content/", "file://")):
        raise ValueError("A filesystem path entered the analytics artifact.")
    for table in analytics["bivariate_bins"].values():
        if any(
            cell["count"] < SUPPRESSION_THRESHOLD
            for cell in table["published_cells"]
        ):
            raise ValueError("A bivariate cell violated the suppression threshold.")


def canonical_json_bytes(analytics: dict[str, Any]) -> bytes:
    """Serialize with stable ordering, indentation, and a terminal newline."""
    return (
        json.dumps(
            analytics,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_artifact(
    analytics: dict[str, Any],
    *,
    output_path: Path,
    sidecar_path: Path,
) -> str:
    payload = canonical_json_bytes(analytics)
    digest = hashlib.sha256(payload).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    sidecar_path.write_text(f"{digest}\n", encoding="ascii", newline="\n")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_dashboard_analytics",
        description=__doc__,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR_PATH)
    parser.add_argument(
        "--generated-at-utc",
        help="Fixed ISO-8601 timestamp; SOURCE_DATE_EPOCH is also supported.",
    )
    parser.add_argument(
        "--generator-commit",
        required=True,
        help="Frozen full commit hash containing this generator and its tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generator_commit = validate_generator_commit(args.generator_commit)
    except ValueError as error:
        raise SystemExit(
            f"build_dashboard_analytics: error: {error}"
        ) from None

    source_path = args.source.resolve()
    if not source_path.is_file():
        raise FileNotFoundError("The labeled training dataset is unavailable.")
    source_hash = calculate_sha256(source_path)
    if source_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError("The labeled training dataset SHA-256 does not match.")

    labeled_data = pd.read_csv(source_path, low_memory=False)
    analytics = build_analytics(
        labeled_data,
        generated_at_utc=_normalize_timestamp(args.generated_at_utc),
        generator_commit=generator_commit,
        source_dataset_sha256=source_hash,
    )
    digest = write_artifact(
        analytics,
        output_path=args.output.resolve(),
        sidecar_path=args.sidecar.resolve(),
    )
    print(
        json.dumps(
            {
                "artifact_sha256": digest,
                "customer_count": analytics["customer_count"],
                "row_count": analytics["row_count"],
                "source_partition": analytics["source_partition"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
