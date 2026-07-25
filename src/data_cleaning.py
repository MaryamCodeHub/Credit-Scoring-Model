"""Deterministic, leakage-safe cleaning for the Kaggle credit dataset."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


PROTECTED_COLUMNS = ("ID", "Customer_ID", "Month", "Credit_Score")

NUMERIC_COLUMNS = (
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
)

REQUIRED_COLUMNS = (
    *PROTECTED_COLUMNS,
    "Name",
    "SSN",
    "Occupation",
    *NUMERIC_COLUMNS,
    "Type_of_Loan",
    "Credit_Mix",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
)

EXACT_PLACEHOLDERS = {
    "Occupation": "_______",
    "Credit_Mix": "_",
    "Credit_History_Age": "NA",
    "Payment_Behaviour": "!@9#%8",
    "SSN": "#F%$D@*&8",
}

NONNEGATIVE_COLUMNS = (
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
)

LOAN_TYPE_ORDER = (
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
    "Not Specified",
)

MONTHLY_BALANCE_SENTINEL = -3.333333333333333e26

_NUMERIC_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_CREDIT_HISTORY_PATTERN = re.compile(
    r"^(?P<years>\d+)\s+Years?\s+and\s+"
    r"(?P<months>\d+)\s+Months?$",
    flags=re.IGNORECASE,
)
_LOAN_SPLIT_PATTERN = re.compile(r",\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)
_CANONICAL_LOAN_TYPES = {value.casefold(): value for value in LOAN_TYPE_ORDER}
_LOAN_SORT_ORDER = {value: index for index, value in enumerate(LOAN_TYPE_ORDER)}


def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministically cleaned copy of a raw credit DataFrame.

    The cleaner uses fixed parsing and domain rules only. It does not mutate
    ``df``, inspect target statistics, fit learned transformations, perform
    imputation, create a model matrix, or write data to disk.

    All original columns are retained. ``Credit_History_Age_Months`` is added
    while the cleaned source duration remains available for traceability.
    """
    _validate_required_columns(df)
    cleaned = df.copy(deep=True)

    _clean_unprotected_strings(cleaned)
    _replace_exact_placeholders(cleaned)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = _parse_numeric_series(cleaned[column])

    _invalidate_fixed_impossible_values(cleaned)
    cleaned["Credit_History_Age_Months"] = cleaned[
        "Credit_History_Age"
    ].map(_parse_credit_history_age)
    cleaned["Type_of_Loan"] = cleaned["Type_of_Loan"].map(
        _normalize_loan_types
    )

    return cleaned


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns for deterministic cleaning: "
            f"{missing}"
        )


def _clean_unprotected_strings(df: pd.DataFrame) -> None:
    for column in df.columns:
        if column in PROTECTED_COLUMNS:
            continue
        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        values = df[column].astype("string").str.strip()
        df[column] = values.mask(values.eq(""), pd.NA)


def _replace_exact_placeholders(df: pd.DataFrame) -> None:
    for column, placeholder in EXACT_PLACEHOLDERS.items():
        df[column] = df[column].mask(df[column].eq(placeholder), pd.NA)

    df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].mask(
        df["Changed_Credit_Limit"].eq("_"),
        pd.NA,
    )
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].mask(
        df["Payment_of_Min_Amount"].eq("NM"),
        "Unknown",
    )


def _parse_numeric_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    text = values.astype("string").str.strip()
    undecorated = text.str.replace(r"^_+|_+$", "", regex=True)
    valid = undecorated.str.fullmatch(_NUMERIC_PATTERN, na=False)
    return pd.to_numeric(undecorated.where(valid), errors="coerce")


def _invalidate_fixed_impossible_values(df: pd.DataFrame) -> None:
    df.loc[(df["Age"] < 0) | (df["Age"] > 120), "Age"] = np.nan

    for column in NONNEGATIVE_COLUMNS:
        df.loc[df[column] < 0, column] = np.nan

    utilization = df["Credit_Utilization_Ratio"]
    df.loc[
        (utilization < 0) | (utilization > 100),
        "Credit_Utilization_Ratio",
    ] = np.nan

    df.loc[
        df["Monthly_Balance"].eq(MONTHLY_BALANCE_SENTINEL),
        "Monthly_Balance",
    ] = np.nan


def _parse_credit_history_age(value: object) -> float:
    if pd.isna(value):
        return np.nan

    match = _CREDIT_HISTORY_PATTERN.fullmatch(str(value))
    if match is None:
        return np.nan

    years = int(match.group("years"))
    months = int(match.group("months"))
    if months > 11:
        return np.nan
    return float(years * 12 + months)


def _normalize_loan_types(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    tokens = [
        re.sub(r"\s+", " ", token).strip()
        for token in _LOAN_SPLIT_PATTERN.split(str(value))
    ]
    tokens = [token for token in tokens if token]
    if not tokens:
        return pd.NA

    normalized: dict[str, str] = {}
    for token in tokens:
        canonical = _CANONICAL_LOAN_TYPES.get(token.casefold(), token)
        normalized.setdefault(canonical.casefold(), canonical)

    ordered = sorted(
        normalized.values(),
        key=lambda token: (
            _LOAN_SORT_ORDER.get(token, len(_LOAN_SORT_ORDER)),
            token.casefold(),
        ),
    )
    return ", ".join(ordered)
