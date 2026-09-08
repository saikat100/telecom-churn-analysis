# src/cleaner.py
"""
cleaner.py
----------
All data cleaning logic for the telecom churn analysis project.

Every function is stateless — it takes a DataFrame in and returns a
cleaned DataFrame out. No global state, no side effects.

Design principle: every cleaning decision has a documented business
justification in the function's Assumptions section. Technical
correctness alone is not sufficient — the business reason must be clear.

Analytical questions served:
    All four questions depend on clean data. Cleaning errors here
    propagate silently into every downstream analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ── Type Corrections ───────────────────────────────────────────────────────────
def fix_total_charges_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Convert the total_charges column from object to float.

    The source system exports total_charges as a string. Some rows contain
    a blank string or whitespace instead of a numeric value. pd.to_numeric
    with errors='coerce' converts these to NaN for downstream handling.

    Assumptions
    -----------
    - total_charges is intended to be a float representing cumulative billing.
    - Non-numeric values in this column are not valid charge amounts.
    - NaN values created here will be handled by impute_total_charges().

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Cleaned DataFrame and count of values that became NaN after conversion.
    """
    df = df.copy()
    before_null_count = df["total_charges"].isna().sum()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    after_null_count = df["total_charges"].isna().sum()
    newly_null = int(after_null_count - before_null_count)
    return df, newly_null


def fix_senior_citizen_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the senior_citizen column from integer (0/1) to string (No/Yes).

    Business justification: senior_citizen is a categorical attribute, not
    a numeric measure. Storing it as 0/1 causes it to be treated as a
    numeric feature in correlation analysis and distance calculations,
    producing misleading results. Standardising to Yes/No aligns it with
    all other binary categorical columns in the dataset.

    Assumptions
    -----------
    - Only values 0 and 1 exist in this column. Any unexpected value
      will map to NaN via the mapping dict, which will surface in profiling.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})
    return df


# ── Missing Value Imputation ───────────────────────────────────────────────────
def impute_total_charges(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Impute missing values in total_charges using a two-pass strategy.

    Pass 1 — Business rule imputation:
        Customers with tenure_months == 0 have not yet completed a billing
        cycle. Their total_charges should logically be 0.0, not missing.
        This is structured missingness with a known business cause.

    Pass 2 — Median imputation:
        Any remaining NaN values after Pass 1 are imputed using the column
        median. Median is preferred over mean because monthly_charges
        distributions in telecom are typically right-skewed, making the
        median a more robust central tendency estimate.

    Assumptions
    -----------
    - tenure_months == 0 always implies no completed billing cycle.
    - Remaining missing values (non-zero tenure) are missing at random
      and do not represent a structured data quality issue.
    - fix_total_charges_type() has already been called before this function.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Cleaned DataFrame and an audit dict with imputation counts per pass.
    """
    df = df.copy()
    audit = {"pass1_business_rule": 0, "pass2_median": 0, "median_used": None}

    # Pass 1: tenure == 0 → total_charges = 0.0
    pass1_mask = (df["total_charges"].isna()) & (df["tenure_months"] == 0)
    audit["pass1_business_rule"] = int(pass1_mask.sum())
    df.loc[pass1_mask, "total_charges"] = 0.0

    # Pass 2: remaining NaN → median
    remaining_mask = df["total_charges"].isna()
    if remaining_mask.sum() > 0:
        median_val = df["total_charges"].median()
        audit["pass2_median"] = int(remaining_mask.sum())
        audit["median_used"] = round(float(median_val), 2)
        df.loc[remaining_mask, "total_charges"] = median_val

    return df, audit


# ── Missing Value Imputation (Categorical Columns) ────────────────────────────
# src/cleaner.py — updated impute_categorical_missing only

def impute_categorical_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Impute missing values in categorical columns using mode (most frequent value)
    imputation.

    Business justification: dependents and payment_method are categorical
    attributes with no natural numeric ordering, so mean/median imputation
    is not applicable. The mode represents the most common customer behavior
    and is the standard approach for categorical missing data when no
    business rule (like the tenure=0 case for total_charges) applies.

    Assumptions
    -----------
    - Missing values in these columns are missing at random and do not
      represent a structured data quality issue tied to another column.
    - The mode is a reasonable default — it does not bias the column's
      existing distribution, since it reinforces the already-dominant
      category rather than introducing a new one.
    - Not every DataFrame passed to this function will contain both
      target columns (e.g. minimal test fixtures) — columns absent from
      the DataFrame are skipped rather than raising a KeyError.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Cleaned DataFrame and an audit dict with imputation counts and
        the mode value used per column.
    """
    df = df.copy()
    audit = {}

    categorical_cols = ["dependents", "payment_method"]

    for col in categorical_cols:
        if col not in df.columns:
            audit[col] = {"rows_imputed": 0, "mode_used": None, "column_present": False}
            continue

        missing_mask = df[col].isna()
        missing_count = int(missing_mask.sum())

        if missing_count > 0:
            mode_val = df[col].mode(dropna=True)[0]
            df.loc[missing_mask, col] = mode_val
            audit[col] = {
                "rows_imputed": missing_count,
                "mode_used": mode_val,
                "column_present": True,
            }
        else:
            audit[col] = {"rows_imputed": 0, "mode_used": None, "column_present": True}

    return df, audit

# ── Consistency Fixes ──────────────────────────────────────────────────────────
def strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from all object (string) columns.

    Business justification: whitespace differences cause value_counts() and
    groupby() to treat 'Yes' and 'Yes ' as different categories, silently
    splitting what should be one group into two. This produces incorrect
    churn rates per segment.

    Assumptions
    -----------
    - All string columns should have no leading or trailing whitespace
      in a well-formed CRM export.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise all column names to lowercase with underscores.

    Prevents case-sensitivity bugs when column names are referenced
    as strings across multiple modules.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    return df



# ── Master Cleaner ─────────────────────────────────────────────────────────────
def run_full_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Execute all cleaning steps in the correct order and return the
    cleaned DataFrame alongside a full audit trail of every action taken.

    Order matters: type fixes must precede imputation; whitespace stripping
    must precede any string comparison or groupby operations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by data_loader.load_raw_data().

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Cleaned DataFrame and a nested audit dict with per-step results.
    """
    audit = {}

    df = standardise_column_names(df)
    audit["columns_standardised"] = True

    df = strip_string_whitespace(df)
    audit["whitespace_stripped"] = True

    df, type_audit_count = fix_total_charges_type(df)
    audit["total_charges_type_fix"] = {"newly_null_created": type_audit_count}

    df = fix_senior_citizen_encoding(df)
    audit["senior_citizen_encoding_fixed"] = True

    df, impute_audit = impute_total_charges(df)
    audit["total_charges_imputation"] = impute_audit

    df, categorical_impute_audit = impute_categorical_missing(df)
    audit["categorical_imputation"] = categorical_impute_audit

    df, dupe_audit = remove_duplicates(df, id_column="customer_id")
    audit["duplicate_removal"] = dupe_audit

    return df, audit


# ── Duplicate Removal ──────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame,
                      id_column: str = "customer_id") -> Tuple[pd.DataFrame, dict]:
    """
    Remove duplicate records using a two-pass strategy.

    Pass 1 — Full row duplicates:
        Rows where every column is identical are dropped. These are
        typically caused by double-exports from the source CRM system.
        First occurrence is kept.

    Pass 2 — ID-level duplicates:
        Rows sharing the same customer_id are dropped, keeping the first
        occurrence. This is a data integrity issue — one customer entity
        should have exactly one record. Keeping the first occurrence is a
        conservative default; in production, the most recent record by
        timestamp would be preferred, but no timestamp exists here.

    Assumptions
    -----------
    - Duplicate rows are export artifacts, not genuine repeated events.
    - First occurrence of a duplicate customer_id is the correct record
      in the absence of a timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
    id_column : str

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Cleaned DataFrame and audit dict with rows dropped per pass.
    """
    df = df.copy()
    audit = {"full_row_duplicates_removed": 0, "id_duplicates_removed": 0}

    # Pass 1: full row duplicates
    before = len(df)
    df = df.drop_duplicates(keep="first")
    audit["full_row_duplicates_removed"] = before - len(df)

    # Pass 2: ID-level duplicates
    before = len(df)
    if id_column in df.columns:
        df = df.drop_duplicates(subset=id_column, keep="first")
    audit["id_duplicates_removed"] = before - len(df)

    return df, audit