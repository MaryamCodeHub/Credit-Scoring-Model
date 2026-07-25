"""Leakage-safe customer-grouped dataset splitting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RANDOM_STATE


@dataclass(frozen=True)
class CustomerGroupedSplit:
    """Three disjoint DataFrame partitions grouped by customer."""

    development_train: pd.DataFrame
    validation: pd.DataFrame
    final_test: pd.DataFrame


def split_by_customer(
    df: pd.DataFrame,
    *,
    group_column: str = "Customer_ID",
    target_column: str = "Credit_Score",
    n_splits: int = 7,
    random_state: int = RANDOM_STATE,
) -> CustomerGroupedSplit:
    """Split labeled rows with scikit-learn's grouped stratification.

    StratifiedGroupKFold uses the target only for class balancing and keeps
    every customer entirely within one fold. One fold is reserved for
    validation, one for the untouched final test, and the remaining folds form
    the development-training set. With seven folds, the approximate
    proportions are 71.4%, 14.3%, and 14.3%.

    The returned DataFrames retain every original column. This function does
    not create a model matrix, transform features, or write any data to disk.
    """
    required_columns = {group_column, target_column}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Missing required columns for customer-grouped splitting: "
            f"{missing_columns}"
        )
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if n_splits < 3:
        raise ValueError("n_splits must be at least 3 to create three partitions.")
    if df[group_column].isna().any():
        raise ValueError(f"{group_column} must not contain missing values.")
    if df[target_column].isna().any():
        raise ValueError(f"{target_column} must not contain missing values.")

    unique_customer_count = df[group_column].nunique()
    if unique_customer_count < n_splits:
        raise ValueError(
            f"At least {n_splits} unique customers are required; "
            f"found {unique_customer_count}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_indices = [
        held_out_positions
        for _, held_out_positions in splitter.split(
            X=df,
            y=df[target_column],
            groups=df[group_column],
        )
    ]

    final_test_positions = np.sort(fold_indices[0])
    validation_positions = np.sort(fold_indices[1])
    development_positions = np.sort(np.concatenate(fold_indices[2:]))

    position_sets = [
        set(development_positions),
        set(validation_positions),
        set(final_test_positions),
    ]
    if any(position_sets[i] & position_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("A row was assigned to more than one partition.")
    if set.union(*position_sets) != set(range(len(df))):
        raise RuntimeError("Not every input row was assigned exactly once.")

    result = CustomerGroupedSplit(
        development_train=df.iloc[development_positions].copy(),
        validation=df.iloc[validation_positions].copy(),
        final_test=df.iloc[final_test_positions].copy(),
    )
    _validate_customer_isolation(result, group_column)
    _validate_class_coverage(result, target_column)
    return result


def _validate_customer_isolation(
    split: CustomerGroupedSplit,
    group_column: str,
) -> None:
    customer_sets = [
        set(split.development_train[group_column]),
        set(split.validation[group_column]),
        set(split.final_test[group_column]),
    ]
    if any(customer_sets[i] & customer_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("A customer was assigned to more than one partition.")


def _validate_class_coverage(
    split: CustomerGroupedSplit,
    target_column: str,
) -> None:
    expected_classes = set(
        pd.concat(
            [
                split.development_train[target_column],
                split.validation[target_column],
                split.final_test[target_column],
            ],
            ignore_index=True,
        )
    )
    for partition_name, partition in (
        ("development_train", split.development_train),
        ("validation", split.validation),
        ("final_test", split.final_test),
    ):
        missing_classes = sorted(expected_classes - set(partition[target_column]))
        if missing_classes:
            raise ValueError(
                f"{partition_name} is missing target classes: {missing_classes}. "
                "Use more groups per class or revise the split configuration."
            )
