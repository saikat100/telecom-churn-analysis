"""
profiler.py
-----------
Reusable data profiling functions for the telecom churn analysis project.
All functions accept a DataFrame and return structured results (DataFrame or dict)
so outputs can be displayed, saved, or tested programmatically.

Analytical questions served:
    - Prerequisite for Q1, Q2, Q3, Q4 — data must be understood before analysis.
"""

import pandas as pd
import numpy as np

# ── Type Audit ─────────────────────────────────────────────────────────────────
def audit_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit each column's stored dtype, unique value count, and a sample value.
    Flags columns whose stored dtype may not match their intended semantic type.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per column with dtype, unique count, sample value, and a flag.
    """
    records = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
        stored_type = str(df[col].dtype)

        is_suspicious = False
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            pct_convertible = converted.notna().mean()
            if pct_convertible > 0.90:
                is_suspicious = True

        records.append({
            "column": col,
            "stored_dtype": stored_type,
            "n_unique": n_unique,
            "sample_value": sample,
            "dtype_suspicious": is_suspicious,
        })

    return pd.DataFrame(records)


# ── Missing Value Report ───────────────────────────────────────────────────────
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a structured missing value report for every column.
    Includes both explicit NaN counts and a check for blank strings,
    which are a common form of hidden missingness in CSV exports.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per column with NaN count, blank string count,
        combined missing count, and percentage of total rows.
        Sorted by combined missing count descending.
    """
    n_rows = len(df)
    records = []

    for col in df.columns:
        nan_count = df[col].isna().sum()

        blank_count = 0
        if df[col].dtype == object:
            blank_count = (df[col].str.strip() == "").sum()

        combined = nan_count + blank_count
        pct = round((combined / n_rows) * 100, 2) if n_rows > 0 else 0.0

        records.append({
            "column": col,
            "nan_count": nan_count,
            "blank_string_count": blank_count,
            "combined_missing": combined,
            "missing_pct": pct,
        })

    result = pd.DataFrame(records).sort_values("combined_missing", ascending=False)
    return result.reset_index(drop=True)


# ── Duplicate Report ───────────────────────────────────────────────────────────
def duplicate_report(df: pd.DataFrame, id_column: str = "customer_id") -> dict:
    """
    Report on full row duplicates and ID-level duplicates separately.

    Full row duplicates — typically accidental double-exports from source system.
    ID duplicates — data integrity issue; one entity has multiple records.

    Parameters
    ----------
    df : pd.DataFrame
    id_column : str
        Column expected to be a unique identifier. Default: 'customer_id'.

    Returns
    -------
    dict
        full_row_duplicates: int
        id_duplicates: int
        id_column: str
        id_duplicate_examples: pd.DataFrame (up to 5 rows)
    """
    full_dupes = int(df.duplicated().sum())
    id_dupes = 0
    id_dupe_examples = pd.DataFrame()

    if id_column in df.columns:
        id_dupes = int(df[id_column].duplicated().sum())
        if id_dupes > 0:
            duped_ids = df[df[id_column].duplicated(keep=False)]
            id_dupe_examples = duped_ids.head(5)

    return {
        "full_row_duplicates": full_dupes,
        "id_duplicates": id_dupes,
        "id_column": id_column,
        "id_duplicate_examples": id_dupe_examples,
    }


# ── Cardinality Report ─────────────────────────────────────────────────────────
def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise unique value counts per column and classify each column's
    likely semantic type based on cardinality thresholds.

    Classification rules:
        identifier       — unique count equals row count
        binary           — exactly 2 unique values
        low-card         — 3 to 20 unique values
        high-card        — 21 to row_count - 1
        constant         — only 1 unique value

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per column with unique count, uniqueness percentage,
        and inferred semantic type label.
    """
    n_rows = len(df)
    records = []

    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        pct_unique = round((n_unique / n_rows) * 100, 2) if n_rows > 0 else 0.0

        if n_unique == 1:
            label = "constant"
        elif n_unique == n_rows:
            label = "identifier"
        elif n_unique == 2:
            label = "binary"
        elif n_unique <= 20:
            label = "low-cardinality categorical"
        else:
            label = "high-cardinality"

        records.append({
            "column": col,
            "n_unique": n_unique,
            "pct_unique": pct_unique,
            "inferred_type": label,
        })

    return pd.DataFrame(records).sort_values("n_unique", ascending=False).reset_index(drop=True)


# ── Target Distribution ────────────────────────────────────────────────────────
def target_distribution(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
    """
    Summarise the distribution of the target variable.
    Flags class imbalance when the minority class is below 30% of total.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Name of the target column. Default: 'churn'.

    Returns
    -------
    pd.DataFrame
        Value counts and percentages for each class in the target column.
        Includes an imbalance flag.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100
    minority_pct = pcts.min()

    result = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "percentage": pcts.values.round(2),
        "imbalance_flag": ["⚠  minority class" if p == minority_pct and minority_pct < 30 else "" for p in pcts.values],
    })

    return result.reset_index(drop=True)


# ── Master Profile Runner ──────────────────────────────────────────────────────
def run_full_profile(df: pd.DataFrame, id_column: str = "customer_id",
                     target_col: str = "churn") -> dict:
    """
    Run all profiling checks in sequence and return results as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
    id_column : str
    target_col : str

    Returns
    -------
    dict
        Keys: shape, dtype_audit, missing_report, duplicate_report,
              cardinality_report, target_distribution
    """
    return {
        "shape": {"n_rows": df.shape[0], "n_columns": df.shape[1]},
        "dtype_audit": audit_dtypes(df),
        "missing_report": missing_value_report(df),
        "duplicate_report": duplicate_report(df, id_column=id_column),
        "cardinality_report": cardinality_report(df),
        "target_distribution": target_distribution(df, target_col=target_col),
    }
