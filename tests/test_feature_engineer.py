"""
Unit tests for src/feature_engineer.py.
Run from project root with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.feature_engineer import (
    add_tenure_group,
    add_total_addon_services,
    add_avg_charge_per_month,
    add_is_new_customer,
    add_has_any_streaming,
    add_is_fully_automated_payer,
    get_feature_hypothesis_registry,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Representative sample covering edge cases for all feature functions."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "tenure_months": [0, 3, 13, 36, 60],
        "monthly_charges": [29.85, 56.95, 53.85, 42.30, 89.10],
        "total_charges": [0.0, 170.85, 700.5, 1522.8, 5346.0],
        "streaming_tv": ["No", "Yes", "No internet service", "Yes", "No"],
        "streaming_movies": ["No", "No", "No internet service", "Yes", "Yes"],
        "online_security": ["No", "Yes", "No", "Yes", "No"],
        "online_backup": ["No", "No", "No", "Yes", "Yes"],
        "device_protection": ["No", "No", "No", "No", "Yes"],
        "tech_support": ["No", "No", "No", "Yes", "Yes"],
        "payment_method": [
            "Electronic check",
            "Bank transfer (automatic)",
            "Mailed check",
            "Credit card (automatic)",
            "Electronic check",
        ],
        "churn": ["No", "Yes", "No", "No", "No"],
    })


# ── Tests: add_tenure_group ────────────────────────────────────────────────────
def test_tenure_group_no_nulls(sample_df):
    result = add_tenure_group(sample_df)
    assert result["tenure_group"].isna().sum() == 0


def test_tenure_group_correct_categories(sample_df):
    result = add_tenure_group(sample_df)
    expected = {"New (0-12m)", "Mid-term (13-24m)", "Established (25-48m)", "Loyal (49m+)"}
    actual = set(result["tenure_group"].dropna().unique())
    assert actual.issubset(expected)


def test_tenure_group_zero_tenure_is_new(sample_df):
    result = add_tenure_group(sample_df)
    assert str(result.iloc[0]["tenure_group"]) == "New (0-12m)"


def test_tenure_group_sixty_months_is_loyal(sample_df):
    result = add_tenure_group(sample_df)
    assert str(result.iloc[4]["tenure_group"]) == "Loyal (49m+)"


def test_tenure_group_is_ordered_categorical(sample_df):
    result = add_tenure_group(sample_df)
    assert result["tenure_group"].cat.ordered


# ── Tests: add_avg_charge_per_month ───────────────────────────────────────────
def test_avg_charge_per_month_no_nulls(sample_df):
    result = add_avg_charge_per_month(sample_df)
    assert result["avg_charge_per_month"].isna().sum() == 0


def test_avg_charge_per_month_no_negatives(sample_df):
    result = add_avg_charge_per_month(sample_df)
    assert (result["avg_charge_per_month"] >= 0).all()


def test_avg_charge_per_month_zero_tenure_uses_monthly(sample_df):
    result = add_avg_charge_per_month(sample_df)
    assert result.iloc[0]["avg_charge_per_month"] == pytest.approx(29.85, rel=1e-2)


def test_avg_charge_per_month_nonzero_tenure_correct(sample_df):
    result = add_avg_charge_per_month(sample_df)
    assert result.iloc[4]["avg_charge_per_month"] == pytest.approx(89.1, rel=1e-2)


# ── Tests: add_is_new_customer ─────────────────────────────────────────────────
def test_is_new_customer_binary(sample_df):
    result = add_is_new_customer(sample_df)
    assert result["is_new_customer"].isin([0, 1]).all()


def test_is_new_customer_zero_tenure_flagged(sample_df):
    result = add_is_new_customer(sample_df, threshold_months=3)
    assert result.iloc[0]["is_new_customer"] == 1
    assert result.iloc[1]["is_new_customer"] == 1


def test_is_new_customer_beyond_threshold_not_flagged(sample_df):
    result = add_is_new_customer(sample_df, threshold_months=3)
    assert result.iloc[2]["is_new_customer"] == 0


def test_is_new_customer_threshold_parameterised(sample_df):
    result = add_is_new_customer(sample_df, threshold_months=12)
    assert result.iloc[0]["is_new_customer"] == 1
    assert result.iloc[2]["is_new_customer"] == 0


# ── Tests: add_has_any_streaming ──────────────────────────────────────────────
def test_has_any_streaming_binary(sample_df):
    result = add_has_any_streaming(sample_df)
    assert result["has_any_streaming"].isin([0, 1]).all()


def test_has_any_streaming_tv_only_flagged(sample_df):
    result = add_has_any_streaming(sample_df)
    assert result.iloc[1]["has_any_streaming"] == 1


def test_has_any_streaming_both_yes_flagged(sample_df):
    result = add_has_any_streaming(sample_df)
    assert result.iloc[3]["has_any_streaming"] == 1


def test_has_any_streaming_no_internet_not_flagged(sample_df):
    result = add_has_any_streaming(sample_df)
    assert result.iloc[2]["has_any_streaming"] == 0


# ── Tests: add_is_fully_automated_payer ───────────────────────────────────────
def test_is_fully_automated_payer_binary(sample_df):
    result = add_is_fully_automated_payer(sample_df)
    assert result["is_fully_automated_payer"].isin([0, 1]).all()


def test_bank_transfer_is_automated(sample_df):
    result = add_is_fully_automated_payer(sample_df)
    assert result.iloc[1]["is_fully_automated_payer"] == 1


def test_credit_card_automatic_is_automated(sample_df):
    result = add_is_fully_automated_payer(sample_df)
    assert result.iloc[3]["is_fully_automated_payer"] == 1


def test_electronic_check_is_not_automated(sample_df):
    result = add_is_fully_automated_payer(sample_df)
    assert result.iloc[0]["is_fully_automated_payer"] == 0


def test_mailed_check_is_not_automated(sample_df):
    result = add_is_fully_automated_payer(sample_df)
    assert result.iloc[2]["is_fully_automated_payer"] == 0


# ── Tests: get_feature_hypothesis_registry ────────────────────────────────────
def test_hypothesis_registry_has_correct_columns():
    registry = get_feature_hypothesis_registry()
    required_cols = {"feature", "source_columns", "business_concept", "hypothesis"}
    assert required_cols.issubset(set(registry.columns))


def test_hypothesis_registry_has_six_features():
    registry = get_feature_hypothesis_registry()
    assert len(registry) == 6
