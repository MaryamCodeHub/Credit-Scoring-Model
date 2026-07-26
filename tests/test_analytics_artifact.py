"""Validation tests for the aggregate-only development analytics artifact."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from scripts import build_dashboard_analytics as analytics_generator
from src.config import ROOT_DIR
from src.extreme_values import EXTREME_VALUE_THRESHOLDS
from src.model_preprocessing import MODEL_FEATURES


ARTIFACT_PATH = (
    ROOT_DIR / "dashboard" / "data" / "development_analytics_v1.json"
)
SIDECAR_PATH = (
    ROOT_DIR / "dashboard" / "data" / "development_analytics_v1.sha256"
)
TRAIN_PATH = (
    ROOT_DIR / "data" / "raw" / "kaggle_credit_score" / "train.csv"
)
TEST_PATH = (
    ROOT_DIR / "data" / "raw" / "kaggle_credit_score" / "test.csv"
)
EXPECTED_TRAIN_SHA256 = (
    "D2EBCC056A64C48710B1AEB96777155835D372D7AD202529F64666011D214DA0"
)
EXPECTED_TEST_SHA256 = (
    "5C606CD0118D49B70D6E934811A0AD806482C2E7F2514FD263315E6BE9DACD9B"
)
FIXED_TIMESTAMP = "2026-07-27T00:00:00Z"
EXPECTED_ROW_COUNT = 71_424
EXPECTED_CUSTOMER_COUNT = 8_928
FORBIDDEN_NAMES = {"ID", "Customer_ID", "Name", "SSN"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def test_schema_and_required_keys(artifact: dict[str, object]) -> None:
    required = {
        "schema_version",
        "generated_at_utc",
        "generator_commit",
        "source_partition",
        "split_seed",
        "source_dataset_sha256",
        "row_count",
        "customer_count",
        "minimum_group_size",
        "privacy",
        "target_distribution",
        "summary_statistics",
        "histograms",
        "occupation_distribution",
        "missing_values",
        "extreme_invalid_counts",
        "correlation_matrix",
        "bivariate_bins",
        "suppressed_bin_count",
    }
    assert required == set(artifact)
    assert artifact["schema_version"] == "1.0.0"
    assert artifact["generated_at_utc"] == FIXED_TIMESTAMP
    assert artifact["source_partition"] == "development_train"
    assert artifact["split_seed"] == 42
    assert artifact["source_dataset_sha256"] == EXPECTED_TRAIN_SHA256


def test_expected_development_size_and_privacy_metadata(
    artifact: dict[str, object],
) -> None:
    assert artifact["row_count"] == EXPECTED_ROW_COUNT
    assert artifact["customer_count"] == EXPECTED_CUSTOMER_COUNT
    assert artifact["minimum_group_size"] == 8
    assert artifact["privacy"] == {
        "aggregate_only": True,
        "forbidden_fields_absent": True,
        "suppression_threshold": 25,
    }


def test_target_counts_and_percentages_reconcile(
    artifact: dict[str, object],
) -> None:
    distribution = artifact["target_distribution"]
    assert set(distribution) == {"Poor", "Standard", "Good"}
    assert sum(item["count"] for item in distribution.values()) == EXPECTED_ROW_COUNT
    percentages = [item["percentage"] for item in distribution.values()]
    assert all(0 <= value <= 100 for value in percentages)
    assert sum(percentages) == pytest.approx(100.0, abs=1e-5)


def test_forbidden_identifiers_pii_paths_and_secrets_are_absent(
    artifact: dict[str, object],
) -> None:
    all_values = list(walk(artifact))
    assert FORBIDDEN_NAMES.isdisjoint(
        value for value in all_values if isinstance(value, str)
    )
    serialized = json.dumps(artifact, sort_keys=True)
    assert ":\\" not in serialized
    assert "/content/" not in serialized
    assert "file://" not in serialized
    assert "api_key" not in serialized.casefold()
    assert "password" not in serialized.casefold()
    assert "credential" not in serialized.casefold()


def test_no_row_level_arrays_or_record_collections(
    artifact: dict[str, object],
) -> None:
    forbidden_collection_keys = {"rows", "records", "raw_rows", "customers"}
    assert forbidden_collection_keys.isdisjoint(
        value for value in walk(artifact) if isinstance(value, str)
    )
    for value in walk(artifact):
        if isinstance(value, list):
            assert len(value) < EXPECTED_CUSTOMER_COUNT


def test_histograms_are_fixed_and_reconcile(
    artifact: dict[str, object],
) -> None:
    histograms = artifact["histograms"]
    assert set(histograms) == set(analytics_generator.HISTOGRAM_EDGES)
    for feature, expected_edges in analytics_generator.HISTOGRAM_EDGES.items():
        histogram = histograms[feature]
        assert histogram["bin_edges"] == list(expected_edges)
        assert len(histogram["counts"]) == len(expected_edges) - 1
        assert len(histogram["percentages"]) == len(histogram["counts"])
        assert sum(histogram["counts"]) + histogram["missing_count"] == EXPECTED_ROW_COUNT
        assert all(0 <= value <= 100 for value in histogram["percentages"])


def test_occupation_and_missing_value_aggregates_reconcile(
    artifact: dict[str, object],
) -> None:
    occupations = artifact["occupation_distribution"]
    assert sum(item["count"] for item in occupations.values()) == EXPECTED_ROW_COUNT
    assert all(
        item["count"] >= 25
        for category, item in occupations.items()
        if category != "Other"
    )
    missing = artifact["missing_values"]
    assert set(missing) == set(MODEL_FEATURES)
    assert all(0 <= item["count"] <= EXPECTED_ROW_COUNT for item in missing.values())
    assert all(0 <= item["percentage"] <= 100 for item in missing.values())


def test_extreme_invalid_counts_use_only_approved_rules(
    artifact: dict[str, object],
) -> None:
    counts = artifact["extreme_invalid_counts"]
    assert set(counts) == set(EXTREME_VALUE_THRESHOLDS)
    for feature, threshold in EXTREME_VALUE_THRESHOLDS.items():
        assert counts[feature]["threshold"] == threshold
        assert 0 <= counts[feature]["count"] <= EXPECTED_ROW_COUNT


def test_correlation_matrix_is_labeled_square_and_symmetric(
    artifact: dict[str, object],
) -> None:
    correlation = artifact["correlation_matrix"]
    features = correlation["features"]
    values = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in correlation["values"]
        ],
        dtype=float,
    )
    assert correlation["method"] == "pearson_pairwise_complete"
    assert features == list(analytics_generator.INPUT_NUMERICAL_FEATURES)
    assert values.shape == (len(features), len(features))
    assert np.allclose(values, values.T, equal_nan=True, atol=1e-6)


def test_bivariate_cells_are_suppressed_and_reconcile(
    artifact: dict[str, object],
) -> None:
    total_suppressed_cells = 0
    for table in artifact["bivariate_bins"].values():
        published_count = sum(cell["count"] for cell in table["published_cells"])
        assert all(cell["count"] >= 25 for cell in table["published_cells"])
        assert (
            published_count
            + table["suppressed_observation_count"]
            + table["missing_pair_count"]
            == EXPECTED_ROW_COUNT
        )
        total_suppressed_cells += table["suppressed_cell_count"]
    assert artifact["suppressed_bin_count"] == total_suppressed_cells


def test_sidecar_matches_exact_json_bytes() -> None:
    expected = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    assert SIDECAR_PATH.read_text(encoding="ascii").strip() == expected


def test_missing_generator_commit_argument_is_rejected_without_internal_details(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "build_dashboard_analytics.py"),
            "--generated-at-utc",
            FIXED_TIMESTAMP,
            "--output",
            str(tmp_path / "unused.json"),
            "--sidecar",
            str(tmp_path / "unused.sha256"),
        ],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "--generator-commit" in result.stderr
    assert str(ROOT_DIR) not in result.stderr
    assert "Traceback" not in result.stderr


def test_malformed_generator_commit_is_rejected() -> None:
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        analytics_generator.validate_generator_commit("not-a-full-commit")


def test_nonexistent_generator_commit_is_rejected_without_git_details() -> None:
    with pytest.raises(ValueError, match="does not exist locally") as error:
        analytics_generator.validate_generator_commit("0" * 40)
    assert str(ROOT_DIR) not in str(error.value)


def test_valid_frozen_generator_commit_is_accepted(
    artifact: dict[str, object],
) -> None:
    frozen_commit = artifact["generator_commit"]
    assert analytics_generator.validate_generator_commit(frozen_commit) == frozen_commit


def test_explicit_frozen_commit_regeneration_is_byte_identical_after_head_changes(
    tmp_path: Path,
    artifact: dict[str, object],
) -> None:
    before = {TRAIN_PATH: file_hash(TRAIN_PATH), TEST_PATH: file_hash(TEST_PATH)}
    frozen_commit = artifact["generator_commit"]
    output_path = tmp_path / "analytics.json"
    sidecar_path = tmp_path / "analytics.sha256"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "build_dashboard_analytics.py"),
            "--generated-at-utc",
            FIXED_TIMESTAMP,
            "--generator-commit",
            frozen_commit,
            "--output",
            str(output_path),
            "--sidecar",
            str(sidecar_path),
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert '"source_partition": "development_train"' in result.stdout
    regenerated = json.loads(output_path.read_text(encoding="utf-8"))
    assert regenerated["generator_commit"] == frozen_commit
    assert output_path.read_bytes() == ARTIFACT_PATH.read_bytes()
    assert sidecar_path.read_bytes() == SIDECAR_PATH.read_bytes()
    assert file_hash(TRAIN_PATH) == before[TRAIN_PATH] == EXPECTED_TRAIN_SHA256
    assert file_hash(TEST_PATH) == before[TEST_PATH] == EXPECTED_TEST_SHA256


def test_generator_never_accesses_held_out_features_or_model_operations() -> None:
    source = Path(analytics_generator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    prohibited_partition_attributes = {"validation", "final_test"}
    accessed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert prohibited_partition_attributes.isdisjoint(accessed_attributes)

    prohibited_calls = {"fit", "fit_transform", "predict", "predict_proba", "score"}
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert prohibited_calls.isdisjoint(called_attributes)
