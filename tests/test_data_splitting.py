"""Tests for leakage-safe customer-grouped splitting."""

import hashlib
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_splitting import split_by_customer


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
]
TARGET_CLASSES = ["Poor", "Standard", "Good"]


@pytest.fixture
def monthly_customer_data() -> pd.DataFrame:
    """Create 21 customers with eight monthly rows and balanced classes."""
    rows = []
    for customer_number in range(21):
        target = TARGET_CLASSES[customer_number % len(TARGET_CLASSES)]
        for month_number, month in enumerate(MONTHS):
            rows.append(
                {
                    "ID": f"row-{customer_number}-{month_number}",
                    "Customer_ID": f"customer-{customer_number}",
                    "Month": month,
                    "Name": f"name-{customer_number}",
                    "SSN": f"ssn-{customer_number}",
                    "Credit_Score": target,
                }
            )
    return pd.DataFrame(rows)


def _customer_sets(split):
    return [
        set(split.development_train["Customer_ID"]),
        set(split.validation["Customer_ID"]),
        set(split.final_test["Customer_ID"]),
    ]


def test_no_customer_overlap(monthly_customer_data):
    split = split_by_customer(monthly_customer_data)
    development, validation, final_test = _customer_sets(split)

    assert development.isdisjoint(validation)
    assert development.isdisjoint(final_test)
    assert validation.isdisjoint(final_test)


def test_all_rows_are_retained_once(monthly_customer_data):
    split = split_by_customer(monthly_customer_data)
    combined_ids = pd.concat(
        [
            split.development_train["ID"],
            split.validation["ID"],
            split.final_test["ID"],
        ],
        ignore_index=True,
    )

    assert len(combined_ids) == len(monthly_customer_data)
    assert combined_ids.is_unique
    assert set(combined_ids) == set(monthly_customer_data["ID"])


def test_same_seed_produces_identical_assignments(monthly_customer_data):
    first = split_by_customer(monthly_customer_data, random_state=42)
    second = split_by_customer(monthly_customer_data, random_state=42)

    for first_partition, second_partition in (
        (first.development_train, second.development_train),
        (first.validation, second.validation),
        (first.final_test, second.final_test),
    ):
        assert first_partition.index.tolist() == second_partition.index.tolist()


def test_all_months_for_each_customer_stay_together(monthly_customer_data):
    split = split_by_customer(monthly_customer_data)

    for partition in (
        split.development_train,
        split.validation,
        split.final_test,
    ):
        monthly_counts = partition.groupby("Customer_ID")["Month"].nunique()
        assert (monthly_counts == len(MONTHS)).all()


def test_all_target_classes_are_represented(monthly_customer_data):
    split = split_by_customer(monthly_customer_data)

    for partition in (
        split.development_train,
        split.validation,
        split.final_test,
    ):
        assert set(partition["Credit_Score"]) == set(TARGET_CLASSES)


def test_five_one_one_fold_proportions(monthly_customer_data):
    split = split_by_customer(monthly_customer_data)
    customer_counts = [
        partition["Customer_ID"].nunique()
        for partition in (
            split.development_train,
            split.validation,
            split.final_test,
        )
    ]

    assert customer_counts == [15, 3, 3]


@pytest.mark.parametrize("missing_column", ["Customer_ID", "Credit_Score"])
def test_missing_required_column_has_clear_error(
    monthly_customer_data,
    missing_column,
):
    incomplete = monthly_customer_data.drop(columns=[missing_column])

    with pytest.raises(ValueError, match=missing_column):
        split_by_customer(incomplete)


def test_source_csv_is_not_modified(monthly_customer_data, tmp_path):
    raw_path = tmp_path / "train.csv"
    monthly_customer_data.to_csv(raw_path, index=False)
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    loaded = pd.read_csv(raw_path)
    split_by_customer(loaded)

    after_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert after_hash == before_hash
