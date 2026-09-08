# src/feature_engineer.py
"""
feature_engineer.py
-------------------
All feature engineering logic for the telecom churn analysis project.
Every function creates one or more new columns from existing columns.
No row is removed, no existing column is modified.

Each function documents:
    - The business concept it represents
    - The hypothesis it encodes
    - The source columns it depends on

Analytical questions served:
    Q1 (who is churning)  : tenure_group, is_new_customer
    Q2 (when do they churn): tenure_group, is_new_customer
    Q3 (behavioral signals): total_addon_services, has_any_streaming,
                             is_fully_automated_payer, avg_charge_per_month
    Q4 (risk concentration): all features contribute to segment definition
"""

import pandas as pd
import numpy as np


# ── Feature 1: Tenure Group ────────────────────────────────────────────────────
def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket tenure_months into four customer lifecycle stage categories.

    Business concept: Customer lifecycle stage.
    Hypothesis: Churn risk is not linear with tenure. It peaks early
    (onboarding risk), drops through the mid-term, and reaches its
    lowest point for loyal customers. Bucketing exposes this non-linearity
    more clearly than a raw numeric column.

    Categories
    ----------
    New        : 0–12 months (first year — highest onboarding churn risk)
    Mid-term   : 13–24 months (second year — settling period)
    Established: 25–48 months (years 3–4 — lower churn, higher engagement)
    Loyal      : 49+ months (4+ years — most committed customers)

    Source columns: tenure_months
    New column    : tenure_group (ordered categorical)
    Assumptions
    -----------
    - Bucket boundaries are informed by typical telecom contract renewal cycles
      (1-year and 2-year contracts) and standard industry definitions of
      customer lifecycle stages.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    bins = [-1, 12, 24, 48, float("inf")]
    labels = ["New (0-12m)", "Mid-term (13-24m)", "Established (25-48m)", "Loyal (49m+)"]
    df["tenure_group"] = pd.Categorical(
        pd.cut(df["tenure_months"], bins=bins, labels=labels),
        categories=labels,
        ordered=True,
    )
    return df


# ── Feature 2: Total Add-on Services ──────────────────────────────────────────
def add_total_addon_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of active add-on services per customer.

    Business concept: Service engagement depth.
    Hypothesis: Customers using more add-on services are more deeply
    embedded in the provider's ecosystem, increasing the switching cost
    and reducing the likelihood of churn.

    Add-on services counted: online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies.
    Only rows with value 'Yes' are counted — 'No' and
    'No internet service' / 'No phone service' are treated as 0.

    Source columns: the six add-on service columns listed above.
    New column    : total_addon_services (int, range 0–6)

    Assumptions
    -----------
    - 'No internet service' and 'No phone service' values mean the
      customer does not have the add-on, equivalent to 'No'.
    - All six service columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    addon_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ]
    df["total_addon_services"] = df[addon_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    ).astype(int)
    return df


# ── Feature 3: Average Charge Per Month ───────────────────────────────────────
def add_avg_charge_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the average amount charged per month over a customer's tenure.

    Business concept: Effective spend rate.
    Hypothesis: Customers whose effective monthly spend is significantly
    higher than the median may be more price-sensitive and therefore at
    higher churn risk, particularly if they are on short-term contracts
    with frequent opportunities to switch providers.

    Formula: total_charges / tenure_months
    For tenure_months == 0, the current monthly_charges is used as the
    best available estimate (customer has not yet been billed).

    Source columns: total_charges, tenure_months, monthly_charges
    New column    : avg_charge_per_month (float)

    Assumptions
    -----------
    - total_charges has already been imputed (no NaN values).
    - For new customers (tenure = 0), monthly_charges is a reasonable
      proxy for their expected ongoing spend.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["avg_charge_per_month"] = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / df["tenure_months"],
        df["monthly_charges"],
    )
    df["avg_charge_per_month"] = df["avg_charge_per_month"].round(2)
    return df


# ── Feature 4: Is New Customer Flag ───────────────────────────────────────────
def add_is_new_customer(df: pd.DataFrame,
                        threshold_months: int = 3) -> pd.DataFrame:
    """
    Binary flag indicating whether a customer is within their first
    N months (default: 3 months).

    Business concept: Early churn risk window.
    Hypothesis: The first 3 months represent the highest-risk period
    in the customer lifecycle. Customers who churn early typically do
    so because of onboarding friction, unmet expectations, or a better
    competitor offer encountered immediately after signing up.
    Flagging this group enables targeted early-intervention analysis.

    Source columns: tenure_months
    New column    : is_new_customer (int, 1 = new, 0 = not new)

    Assumptions
    -----------
    - 3 months is the industry-standard early-churn threshold in telecom.
      This threshold is parameterised so it can be adjusted without
      modifying the function body.

    Parameters
    ----------
    df : pd.DataFrame
    threshold_months : int
        Customers with tenure <= this value are flagged as new.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["is_new_customer"] = (df["tenure_months"] <= threshold_months).astype(int)
    return df


# ── Feature 5: Has Any Streaming ──────────────────────────────────────────────
def add_has_any_streaming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag indicating whether a customer subscribes to any
    streaming service (TV or movies).

    Business concept: Entertainment bundle adoption.
    Hypothesis: Customers using streaming services consume more of the
    provider's infrastructure, increasing perceived value and switching
    cost. Bundle adopters are expected to show lower churn rates.

    Source columns: streaming_tv, streaming_movies
    New column    : has_any_streaming (int, 1 = at least one streaming service, 0 = none)

    Assumptions
    -----------
    - 'Yes' in either streaming column indicates active streaming usage.
    - 'No internet service' is treated as no streaming (value = 0).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["has_any_streaming"] = (
        (df["streaming_tv"] == "Yes") | (df["streaming_movies"] == "Yes")
    ).astype(int)
    return df


# ── Feature 6: Is Fully Automated Payer ───────────────────────────────────────
def add_is_fully_automated_payer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag indicating whether a customer uses a fully automated
    payment method (bank transfer or credit card automatic).

    Business concept: Payment friction and inertia.
    Hypothesis: Customers on automatic payment methods experience less
    billing friction and are less likely to actively review their bill
    each month. This passive engagement reduces the probability of
    deliberate churn triggered by a billing review.

    Source columns: payment_method
    New column    : is_fully_automated_payer (int, 1 = automated, 0 = manual)

    Assumptions
    -----------
    - 'Bank transfer (automatic)' and 'Credit card (automatic)' are
      the two fully automated payment methods in this dataset.
    - 'Electronic check' and 'Mailed check' require active customer
      action each billing cycle and are classified as manual.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    automated_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
    df["is_fully_automated_payer"] = (
        df["payment_method"].isin(automated_methods)
    ).astype(int)
    return df


# ── Feature Hypothesis Registry ───────────────────────────────────────────────
def get_feature_hypothesis_registry() -> pd.DataFrame:
    """
    Return a DataFrame documenting every engineered feature, its source
    columns, the business concept it represents, and the hypothesis
    it encodes. Used for generating documentation and for tracking
    which hypotheses were confirmed or rejected in EDA.

    Returns
    -------
    pd.DataFrame
    """
    registry = [
        {
            "feature": "tenure_group",
            "source_columns": "tenure_months",
            "business_concept": "Customer lifecycle stage",
            "hypothesis": "Churn risk peaks for new customers and declines with tenure",
            "eda_confirmed": "TBD — Class 3",
        },
        {
            "feature": "total_addon_services",
            "source_columns": "online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies",
            "business_concept": "Service engagement depth",
            "hypothesis": "More add-ons = higher switching cost = lower churn",
            "eda_confirmed": "TBD — Class 3",
        },
        {
            "feature": "avg_charge_per_month",
            "source_columns": "total_charges, tenure_months, monthly_charges",
            "business_concept": "Effective spend rate",
            "hypothesis": "Higher effective spend correlates with price sensitivity and higher churn",
            "eda_confirmed": "TBD — Class 3",
        },
        {
            "feature": "is_new_customer",
            "source_columns": "tenure_months",
            "business_concept": "Early churn risk window",
            "hypothesis": "First 3 months represent disproportionately high churn risk",
            "eda_confirmed": "TBD — Class 3",
        },
        {
            "feature": "has_any_streaming",
            "source_columns": "streaming_tv, streaming_movies",
            "business_concept": "Entertainment bundle adoption",
            "hypothesis": "Streaming users have higher switching cost and lower churn",
            "eda_confirmed": "TBD — Class 3",
        },
        {
            "feature": "is_fully_automated_payer",
            "source_columns": "payment_method",
            "business_concept": "Payment friction and inertia",
            "hypothesis": "Auto-pay customers churn less due to reduced active billing engagement",
            "eda_confirmed": "TBD — Class 3",
        },
    ]
    return pd.DataFrame(registry)