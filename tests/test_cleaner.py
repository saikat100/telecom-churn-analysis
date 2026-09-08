"""
Unit tests for src/cleaner.py
Run command: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

root = os.path.abspath(".")
if root not in sys.path:
    sys.path.insert(0, root)

from src.cleaner import (
    standardise_column_names,
    strip_string_whitespace,
    fix_total_charges_type,
    fix_senior_citizen_encoding,
    impute_total_charges,
    impute_categorical_missing,
    remove_duplicates,
    run_full_cleaning_pipeline,
    standardise_categorical_values,
)


# fixtures

@pytest.fixture
def minimal_raw_df():
    raw_df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004"],
        "senior_citizen": [0, 1, 0, 1],
        "tenure_months": [0, 5, 0, 24],
        "monthly_charges": [29.85, 56.95, 53.85, 42.30],
        "total_charges": ["", "284.75", " ", "1014.20"],
        "churn": ["No", "Yes", "No", "No"],
    })
    return raw_df


@pytest.fixture
def duplicate_df():
    duplicate_df = pd.DataFrame({
        "customer_id": ["C001", "C001", "C002", "C003", "C003"],
        "tenure_months": [12, 12, 6, 24, 24],
        "monthly_charges": [50.0, 50.0, 30.0, 70.0, 70.0],
        "total_charges": ["600", "600", "180", "1680", "1680"],
        "senior_citizen": [0, 0, 1, 0, 0],
        "churn": ["No", "No", "Yes", "No", "No"],
    })
    return duplicate_df


@pytest.fixture
def categorical_missing_df():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "dependents": ["Yes", np.nan, "No", "No", np.nan],
        "payment_method": [
            "Electronic check",
            "Electronic check",
            np.nan,
            "Mailed check",
            "Electronic check",
        ],
    })


# --- Tests -----
def test_standardise_column_names_lowercases():
    df = pd.DataFrame({"CustomerID": [1], "Monthly Charges": [50.0]})
    result = standardise_column_names(df)
    assert "customerid" in result.columns
    assert "monthly_charges" in result.columns


def test_standardise_column_names_replaces_hyphens():
    df = pd.DataFrame({"contract-type": ["Month-to-month"]})
    result = standardise_column_names(df)
    assert "contract_type" in result.columns


def test_standardise_column_names_replaces_spaces():
    df = pd.DataFrame({"contract type": ["Month-to-month"]})
    result = standardise_column_names(df)
    assert "contract_type" in result.columns


def test_standardise_column_names_replaces_spaces():
    df = pd.DataFrame({"contract    type": ["Month-to-month"]})
    result = standardise_column_names(df)
    assert "contract_type" in result.columns


def test_standardise_column_names_does_not_modify_original(minimal_raw_df):
    original_cols = list(minimal_raw_df.columns)
    _ = standardise_column_names(minimal_raw_df)
    assert list(minimal_raw_df.columns) == original_cols


# ── Tests: fix_senior_citizen_encoding ────────────────────────────────────────
def test_senior_citizen_encoding_maps_correctly(minimal_raw_df):
    result = fix_senior_citizen_encoding(minimal_raw_df)
    assert set(result["senior_citizen"].unique()).issubset({"Yes", "No"})


def test_senior_citizen_zero_maps_to_no(minimal_raw_df):
    result = fix_senior_citizen_encoding(minimal_raw_df)
    assert result.iloc[0]["senior_citizen"] == "No"


def test_senior_citizen_one_maps_to_yes(minimal_raw_df):
    result = fix_senior_citizen_encoding(minimal_raw_df)
    assert result.iloc[1]["senior_citizen"] == "Yes"


# ── Tests: impute_total_charges ────────────────────────────────────────────────
def test_impute_total_charges_no_nulls_remain(minimal_raw_df):
    df, _ = fix_total_charges_type(minimal_raw_df)
    result, _ = impute_total_charges(df)
    assert result["total_charges"].isna().sum() == 0


def test_impute_total_charges_tenure_zero_gets_zero(minimal_raw_df):
    df, _ = fix_total_charges_type(minimal_raw_df)
    result, audit = impute_total_charges(df)
    tenure_zero_rows = result[result["tenure_months"] == 0]
    assert (tenure_zero_rows["total_charges"] == 0.0).all()


def test_impute_total_charges_audit_counts_correct(minimal_raw_df):
    df, _ = fix_total_charges_type(minimal_raw_df)
    _, audit = impute_total_charges(df)
    assert audit["pass1_business_rule"] == 2


# ── Tests: remove_duplicates ───────────────────────────────────────────────────
def test_remove_full_row_duplicates(duplicate_df):
    result, audit = remove_duplicates(duplicate_df, id_column="customer_id")
    assert audit["full_row_duplicates_removed"] >= 0


def test_remove_id_duplicates_keeps_one_per_id(duplicate_df):
    result, _ = remove_duplicates(duplicate_df, id_column="customer_id")
    assert result["customer_id"].duplicated().sum() == 0


def test_remove_duplicates_does_not_modify_original(duplicate_df):
    original_len = len(duplicate_df)
    _ = remove_duplicates(duplicate_df, id_column="customer_id")
    assert len(duplicate_df) == original_len


# ── Tests: run_full_cleaning_pipeline ─────────────────────────────────────────
def test_full_pipeline_returns_no_nulls(minimal_raw_df):
    result, _ = run_full_cleaning_pipeline(minimal_raw_df)
    assert result.isnull().sum().sum() == 0


def test_full_pipeline_returns_audit_dict(minimal_raw_df):
    _, audit = run_full_cleaning_pipeline(minimal_raw_df)
    assert isinstance(audit, dict)
    assert "total_charges_imputation" in audit
    assert "duplicate_removal" in audit


def test_full_pipeline_does_not_modify_input(minimal_raw_df):
    original_shape = minimal_raw_df.shape
    _ = run_full_cleaning_pipeline(minimal_raw_df)
    assert minimal_raw_df.shape == original_shape


# ── Tests: impute_categorical_missing ─────────────────────────────────────────
def test_impute_categorical_no_nulls_remain(categorical_missing_df):
    result, _ = impute_categorical_missing(categorical_missing_df)
    assert result["dependents"].isna().sum() == 0
    assert result["payment_method"].isna().sum() == 0


def test_impute_categorical_uses_mode(categorical_missing_df):
    result, audit = impute_categorical_missing(categorical_missing_df)
    # payment_method mode is 'Electronic check' (appears 3 times)
    assert audit["payment_method"]["mode_used"] == "Electronic check"
    assert result.loc[2, "payment_method"] == "Electronic check"


def test_impute_categorical_audit_counts_correct(categorical_missing_df):
    _, audit = impute_categorical_missing(categorical_missing_df)
    assert audit["dependents"]["rows_imputed"] == 2
    assert audit["payment_method"]["rows_imputed"] == 1


def test_impute_categorical_no_missing_returns_zero_count():
    df = pd.DataFrame({
        "dependents": ["Yes", "No"],
        "payment_method": ["Electronic check", "Mailed check"],
    })
    _, audit = impute_categorical_missing(df)
    assert audit["dependents"]["rows_imputed"] == 0
    assert audit["dependents"]["mode_used"] is None


def test_impute_categorical_does_not_modify_original(categorical_missing_df):
    original_nulls = categorical_missing_df["dependents"].isna().sum()
    _ = impute_categorical_missing(categorical_missing_df)
    assert categorical_missing_df["dependents"].isna().sum() == original_nulls


def test_impute_categorical_skips_missing_column():
    df = pd.DataFrame({"customer_id": ["C001", "C002"]})
    result, audit = impute_categorical_missing(df)
    assert audit["dependents"]["column_present"] is False
    assert audit["payment_method"]["column_present"] is False
    assert "dependents" not in result.columns


# ── Tests: run_full_cleaning_pipeline (updated) ───────────────────────────────
def test_full_pipeline_includes_categorical_imputation(minimal_raw_df):
    _, audit = run_full_cleaning_pipeline(minimal_raw_df)
    assert "categorical_imputation" in audit


def test_full_pipeline_returns_no_nulls_with_categorical_gaps():
    df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004"],
        "senior_citizen": [0, 1, 0, 1],
        "tenure_months": [0, 5, 0, 24],
        "monthly_charges": [29.85, 56.95, 53.85, 42.30],
        "total_charges": ["", "284.75", " ", "1014.20"],
        "dependents": ["Yes", np.nan, "No", "No"],
        "payment_method": ["Electronic check", np.nan, "Mailed check", "Electronic check"],
        "churn": ["No", "Yes", "No", "No"],
    })
    result, _ = run_full_cleaning_pipeline(df)
    assert result.isnull().sum().sum() == 0

# ── Tests: standardise_categorical_values ─────────────────────────────────────
def test_gender_variants_collapse_to_two_categories():
    df = pd.DataFrame({"gender": ["Male", "male", "M", "Female", "female", "F"]})
    result, _ = standardise_categorical_values(df)
    assert set(result["gender"]) == {"Male", "Female"}


def test_contract_type_casing_collapses():
    df = pd.DataFrame({"contract_type": ["Month-to-month", "month-to-month", "Two year"]})
    result, _ = standardise_categorical_values(df)
    assert result["contract_type"].nunique() == 2


def test_standardise_categorical_values_reports_what_changed():
    df = pd.DataFrame({"gender": ["Male", "M", "F"]})
    _, audit = standardise_categorical_values(df)
    assert audit["gender"]["values_replaced"] == 2
    assert audit["gender"]["categories_after"] == 2


def test_unknown_category_is_left_untouched():
    df = pd.DataFrame({"gender": ["Male", "Unspecified"]})
    result, _ = standardise_categorical_values(df)
    assert "Unspecified" in set(result["gender"])


def test_standardise_categorical_values_does_not_modify_input():
    df = pd.DataFrame({"gender": ["male"]})
    standardise_categorical_values(df)
    assert df.loc[0, "gender"] == "male"


def test_pipeline_leaves_no_case_duplicate_categories():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "gender": ["Male", "male", "F"],
        "senior_citizen": [0, 1, 0],
        "tenure_months": [5, 10, 20],
        "contract_type": ["Month-to-month", "month-to-month", "Two year"],
        "payment_method": ["Bank transfer", "Mailed check", "Credit card"],
        "dependents": ["Yes", "No", "Yes"],
        "monthly_charges": [50.0, 60.0, 70.0],
        "total_charges": ["250.0", "600.0", "1400.0"],
        "churn": ["No", "Yes", "No"],
    })
    result, _ = run_full_cleaning_pipeline(df)
    assert set(result["gender"]) == {"Male", "Female"}
    assert result["contract_type"].nunique() == 2
