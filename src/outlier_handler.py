# src/outlier_handler.py
"""
outlier_handler.py
------------------
Outlier detection and treatment for numeric columns in the telecom
churn dataset. Each column has an explicit, documented treatment strategy.

Three strategies are implemented:
    - remove_invalid_rows : for physically impossible values (errors, not outliers)
    - winsorize           : for plausible but extreme values where capping is preferred
    - flag_outliers       : for genuine extreme values that should be kept but marked

Assumptions
-----------
- run_full_cleaning_pipeline() from cleaner.py has already been applied
  before any function in this module is called.
- Outlier thresholds are defined based on domain knowledge of telecom billing.
  If the data source changes, thresholds should be reviewed.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ── Constants — Explicit Thresholds with Justifications ───────────────────────
# tenure_months: 0–72 is the known range of this dataset (6-year period).
# Values outside this range are data errors, not business outliers.
TENURE_MIN = 0
TENURE_MAX = 72

# monthly_charges: telecom plans in this dataset range from ~18 to ~120.
# Winsorization percentiles chosen to preserve 98% of the distribution.
MONTHLY_CHARGES_LOWER_PCT = 1
MONTHLY_CHARGES_UPPER_PCT = 99


# ── Strategy 1: Remove Invalid Rows ───────────────────────────────────────────
def remove_invalid_tenure_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Remove rows where tenure_months falls outside the valid range [0, 72].

    Business justification: tenure_months represents months as a customer
    in a dataset covering a 6-year period. Negative values or values above
    72 are physically impossible and indicate upstream data entry errors.
    These rows carry no analytical value and should not be imputed.

    Assumptions
    -----------
    - The dataset covers exactly a 6-year observation window (max tenure = 72).
    - No valid business case exists for tenure outside [0, 72].

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Cleaned DataFrame and audit dict with rows removed and their indices.
    """
    df = df.copy()
    invalid_mask = (df["tenure_months"] < TENURE_MIN) | (df["tenure_months"] > TENURE_MAX)
    invalid_count = int(invalid_mask.sum())
    invalid_indices = df[invalid_mask].index.tolist()
    df = df[~invalid_mask].reset_index(drop=True)

    return df, {
        "column": "tenure_months",
        "strategy": "remove_invalid_rows",
        "valid_range": [TENURE_MIN, TENURE_MAX],
        "rows_removed": invalid_count,
        "removed_indices": invalid_indices,
    }


# ── Strategy 2: Winsorization ──────────────────────────────────────────────────
def winsorize_monthly_charges(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Cap monthly_charges at the 1st and 99th percentiles (Winsorization).

    Business justification: monthly_charges at the extreme ends of the
    distribution are more likely to represent data anomalies than genuine
    high-value or low-value customers in this dataset. Capping preserves
    the row while limiting the influence of extreme values on mean-based
    summaries and downstream analysis. Removing these rows would be too
    aggressive — the customer records themselves are valid.

    Assumptions
    -----------
    - The 1st–99th percentile range captures the genuine distribution of
      telecom plan pricing for this customer base.
    - Values outside this range are anomalies, not premium/discount tiers.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with capped values and audit dict with cap boundaries.
    """
    df = df.copy()
    lower_cap = df["monthly_charges"].quantile(MONTHLY_CHARGES_LOWER_PCT / 100)
    upper_cap = df["monthly_charges"].quantile(MONTHLY_CHARGES_UPPER_PCT / 100)

    below_cap = int((df["monthly_charges"] < lower_cap).sum())
    above_cap = int((df["monthly_charges"] > upper_cap).sum())

    df["monthly_charges"] = df["monthly_charges"].clip(lower=lower_cap, upper=upper_cap)

    return df, {
        "column": "monthly_charges",
        "strategy": "winsorization",
        "lower_cap": round(float(lower_cap), 2),
        "upper_cap": round(float(upper_cap), 2),
        "values_capped_below": below_cap,
        "values_capped_above": above_cap,
    }


# ── Strategy 3: Flag Outliers ──────────────────────────────────────────────────
def flag_total_charges_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Flag total_charges outliers with a binary indicator column rather
    than capping or removing them.

    Business justification: high total_charges values naturally arise from
    customers with long tenure and high monthly plans. These are genuine
    long-term, high-value customers — not errors. Removing or capping them
    would destroy the signal that high-value customers may behave differently
    regarding churn. The flag column allows Class 3 EDA to examine this
    group explicitly without distorting the main distribution.

    The IQR method is used for detection:
        lower bound = Q1 - 1.5 * IQR
        upper bound = Q3 + 1.5 * IQR
    Values outside these bounds are flagged as outliers.

    Assumptions
    -----------
    - High total_charges is a business reality for long-tenure customers,
      not a data quality issue.
    - IQR-based thresholds are appropriate for a right-skewed billing column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with new flag column and audit dict with threshold details.
    """
    df = df.copy()
    q1 = df["total_charges"].quantile(0.25)
    q3 = df["total_charges"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 1 = outlier, 0 = within normal range
    df["total_charges_outlier_flag"] = (
        (df["total_charges"] < lower_bound) | (df["total_charges"] > upper_bound)
    ).astype(int)

    flagged_count = int(df["total_charges_outlier_flag"].sum())

    return df, {
        "column": "total_charges",
        "strategy": "flag_outliers",
        "iqr_lower_bound": round(float(lower_bound), 2),
        "iqr_upper_bound": round(float(upper_bound), 2),
        "rows_flagged": flagged_count,
        "flag_column_created": "total_charges_outlier_flag",
    }


# ── Outlier Decision Log ───────────────────────────────────────────────────────
def log_outlier_decisions() -> pd.DataFrame:
    """
    Return a static DataFrame documenting the outlier treatment decision
    for every numeric column, including columns where no action was taken.

    This is the human-readable record of decisions made — designed to be
    saved to docs/ or included in the final report. It answers:
    "why did you treat outliers this way?"

    Returns
    -------
    pd.DataFrame
        One row per numeric column with strategy, justification, and threshold.
    """
    decisions = [
        {
            "column": "tenure_months",
            "strategy": "remove_invalid_rows",
            "threshold": f"Valid range: [{TENURE_MIN}, {TENURE_MAX}]",
            "justification": "Values outside 0–72 are physically impossible given the 6-year observation window. Treated as data errors, not outliers.",
        },
        {
            "column": "monthly_charges",
            "strategy": "winsorization",
            "threshold": f"Capped at {MONTHLY_CHARGES_LOWER_PCT}th and {MONTHLY_CHARGES_UPPER_PCT}th percentile",
            "justification": "Extreme values are likely anomalies rather than genuine premium plans. Capping preserves the row while limiting distortion of mean-based summaries.",
        },
        {
            "column": "total_charges",
            "strategy": "flag_outliers (IQR method)",
            "threshold": "Q1 - 1.5*IQR to Q3 + 1.5*IQR",
            "justification": "High total charges reflect genuine long-tenure, high-value customers. Removing or capping would destroy this signal. Flagged for separate examination in EDA.",
        },
        {
            "column": "avg_charge_per_month",
            "strategy": "no action (derived feature)",
            "threshold": "N/A",
            "justification": "Derived from monthly_charges (already Winsorized) and tenure_months (already validated). Outlier treatment on source columns is sufficient.",
        },
    ]
    return pd.DataFrame(decisions)


# ── Master Outlier Pipeline ────────────────────────────────────────────────────
def run_full_outlier_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Execute all outlier treatment steps in the correct order.

    Order: remove invalids first (reduces dataset size before percentile
    calculations), then Winsorize (percentiles are now stable), then flag.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after run_full_cleaning_pipeline() has been applied.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Treated DataFrame and a combined audit dict from all steps.
    """
    audit = {}

    df, tenure_audit = remove_invalid_tenure_rows(df)
    audit["tenure_months"] = tenure_audit

    df, charges_audit = winsorize_monthly_charges(df)
    audit["monthly_charges"] = charges_audit

    df, flag_audit = flag_total_charges_outliers(df)
    audit["total_charges"] = flag_audit

    return df, audit