"""Tests for deterministic, leakage-safe credit-data cleaning."""

import hashlib
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import clean_credit_data


def _raw_row(**overrides):
    row = {
        "ID": "row_1",
        "Customer_ID": "customer_1",
        "Month": "January",
        "Name": " Example Name ",
        "Age": "30",
        "SSN": "111-22-3333",
        "Occupation": "Media_Manager",
        "Annual_Income": "50000",
        "Monthly_Inhand_Salary": "4000",
        "Num_Bank_Accounts": "3",
        "Num_Credit_Card": "2",
        "Interest_Rate": "8",
        "Num_of_Loan": "1",
        "Type_of_Loan": "Student Loan",
        "Delay_from_due_date": "-2",
        "Num_of_Delayed_Payment": "0",
        "Changed_Credit_Limit": "-1.5",
        "Num_Credit_Inquiries": "1",
        "Credit_Mix": "Standard",
        "Outstanding_Debt": "1000.50",
        "Credit_Utilization_Ratio": "25.5",
        "Credit_History_Age": "2 Years and 3 Months",
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": "250.25",
        "Amount_invested_monthly": "300.75",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": "450.50",
        "Credit_Score": "Good",
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_frame():
    return pd.DataFrame(
        [
            _raw_row(),
            _raw_row(
                ID="row_2",
                Customer_ID="customer_2",
                Month="February",
                Credit_Score="Poor",
            ),
        ],
        index=[10, 20],
    )


def test_input_is_not_mutated(raw_frame):
    original = raw_frame.copy(deep=True)

    clean_credit_data(raw_frame)

    pd.testing.assert_frame_equal(raw_frame, original)


def test_rows_index_and_original_columns_are_preserved(raw_frame):
    cleaned = clean_credit_data(raw_frame)

    assert len(cleaned) == len(raw_frame)
    assert cleaned.index.equals(raw_frame.index)
    assert set(raw_frame.columns).issubset(cleaned.columns)


def test_exact_placeholders_and_blanks_are_handled():
    frame = pd.DataFrame(
        [
            _raw_row(
                Occupation="_______",
                Changed_Credit_Limit="_",
                Credit_Mix="_",
                Credit_History_Age="NA",
                Payment_Behaviour="!@9#%8",
                SSN="#F%$D@*&8",
                Payment_of_Min_Amount="NM",
                Type_of_Loan="   ",
            )
        ]
    )

    cleaned = clean_credit_data(frame).iloc[0]

    for column in (
        "Occupation",
        "Changed_Credit_Limit",
        "Credit_Mix",
        "Credit_History_Age",
        "Payment_Behaviour",
        "SSN",
        "Type_of_Loan",
    ):
        assert pd.isna(cleaned[column])
    assert cleaned["Payment_of_Min_Amount"] == "Unknown"


def test_numeric_underscore_decoration_is_parsed_safely():
    frame = pd.DataFrame(
        [
            _raw_row(
                Age="_42_",
                Annual_Income="50000_",
                Num_of_Loan="_3",
                Outstanding_Debt="_1250.75_",
            ),
            _raw_row(
                ID="row_2",
                Customer_ID="customer_2",
                Annual_Income="12_34",
            ),
        ]
    )

    cleaned = clean_credit_data(frame)

    assert cleaned.loc[0, "Age"] == 42
    assert cleaned.loc[0, "Annual_Income"] == 50000
    assert cleaned.loc[0, "Num_of_Loan"] == 3
    assert cleaned.loc[0, "Outstanding_Debt"] == 1250.75
    assert pd.isna(cleaned.loc[1, "Annual_Income"])


def test_meaningful_internal_underscores_are_preserved():
    frame = pd.DataFrame(
        [
            _raw_row(
                Customer_ID="customer_abc_123",
                Occupation="Media_Manager",
                Payment_Behaviour="Low_spent_Small_value_payments",
            )
        ]
    )

    cleaned = clean_credit_data(frame).iloc[0]

    assert cleaned["Customer_ID"] == "customer_abc_123"
    assert cleaned["Occupation"] == "Media_Manager"
    assert cleaned["Payment_Behaviour"] == (
        "Low_spent_Small_value_payments"
    )


def test_fixed_impossible_values_become_missing():
    frame = pd.DataFrame(
        [
            _raw_row(
                Age="-1",
                Num_Bank_Accounts="-1",
                Num_Credit_Card="-2",
                Num_of_Loan="-3",
                Num_of_Delayed_Payment="-4",
                Num_Credit_Inquiries="-5",
                Credit_Utilization_Ratio="101",
                Monthly_Balance="-3.333333333333333e+26",
            ),
            _raw_row(
                ID="row_2",
                Customer_ID="customer_2",
                Age="121",
                Credit_Utilization_Ratio="-0.1",
            ),
        ]
    )

    cleaned = clean_credit_data(frame)

    for column in (
        "Age",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "Credit_Utilization_Ratio",
        "Monthly_Balance",
    ):
        assert pd.isna(cleaned.loc[0, column])
    assert pd.isna(cleaned.loc[1, "Age"])
    assert pd.isna(cleaned.loc[1, "Credit_Utilization_Ratio"])


def test_uncertain_signed_values_remain_unchanged():
    frame = pd.DataFrame(
        [_raw_row(Delay_from_due_date="-5", Changed_Credit_Limit="-6.25")]
    )

    cleaned = clean_credit_data(frame).iloc[0]

    assert cleaned["Delay_from_due_date"] == -5
    assert cleaned["Changed_Credit_Limit"] == -6.25


def test_credit_history_age_is_converted_to_months():
    frame = pd.DataFrame(
        [
            _raw_row(Credit_History_Age="2 Years and 3 Months"),
            _raw_row(
                ID="row_2",
                Customer_ID="customer_2",
                Credit_History_Age="1 Year and 1 Month",
            ),
            _raw_row(
                ID="row_3",
                Customer_ID="customer_3",
                Credit_History_Age="2 Years and 12 Months",
            ),
        ]
    )

    cleaned = clean_credit_data(frame)

    assert cleaned.loc[0, "Credit_History_Age_Months"] == 27
    assert cleaned.loc[1, "Credit_History_Age_Months"] == 13
    assert pd.isna(cleaned.loc[2, "Credit_History_Age_Months"])
    assert "Credit_History_Age" in cleaned.columns


def test_type_of_loan_is_deduplicated_and_ordered():
    frame = pd.DataFrame(
        [
            _raw_row(
                Type_of_Loan=(
                    "Student Loan, Auto Loan, and Student Loan"
                )
            ),
            _raw_row(
                ID="row_2",
                Customer_ID="customer_2",
                Type_of_Loan="student loan and AUTO LOAN",
            ),
        ]
    )

    cleaned = clean_credit_data(frame)

    assert cleaned.loc[0, "Type_of_Loan"] == "Auto Loan, Student Loan"
    assert cleaned.loc[1, "Type_of_Loan"] == "Auto Loan, Student Loan"


def test_repeated_runs_produce_identical_output(raw_frame):
    first = clean_credit_data(raw_frame)
    second = clean_credit_data(first)

    pd.testing.assert_frame_equal(first, second)


def test_protected_columns_remain_unchanged():
    frame = pd.DataFrame(
        [
            _raw_row(
                ID=" ID_with_spaces ",
                Customer_ID=" Customer_ID_internal ",
                Month=" January ",
                Credit_Score=" Good ",
            )
        ]
    )

    cleaned = clean_credit_data(frame)

    for column in ("ID", "Customer_ID", "Month", "Credit_Score"):
        pd.testing.assert_series_equal(
            cleaned[column],
            frame[column],
            check_names=False,
        )


def test_missing_required_columns_raise_clear_error(raw_frame):
    incomplete = raw_frame.drop(columns=["Age"])

    with pytest.raises(ValueError, match="Age"):
        clean_credit_data(incomplete)


def test_source_csv_is_not_modified(raw_frame, tmp_path):
    raw_path = tmp_path / "train.csv"
    raw_frame.to_csv(raw_path, index=False)
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    loaded = pd.read_csv(raw_path)
    clean_credit_data(loaded)

    after_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert after_hash == before_hash
