"""
data_loader.py
--------------
Handles all raw data loading for the telecom churn analysis project.
Any notebook or script that needs raw data imports from here.

Analytical question served: prerequisite for all four questions —
data must be loaded and structurally validated before any analysis begins.
"""

import os
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",                                     
    "data", "raw", "telecom_customers.csv"
)

EXPECTED_COLUMNS = [
    "customer_id", "gender", "senior_citizen", "partner", "dependents",
    "tenure_months", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract_type", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges", "churn",
]

# ── Functions ──────────────────────────────────────────────────────────────────
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw telecom customer data from a CSV file.
    Performs three structural checks before returning the DataFrame:
    1. File exists at the expected path
    2. File is not empty
    3. All expected columns are present

    Parameters
    ----------
    path : str
        Path to the raw CSV file. Defaults to RAW_DATA_PATH constant.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If no file exists at the given path.
    ValueError
        If the file is empty or required columns are missing.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            f"Place 'telecom_customers.csv' inside the data/raw/ folder."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(
            f"Dataset at '{path}' loaded successfully but contains zero rows."
        )

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_cols)}\n"
            f"Check your data source or update EXPECTED_COLUMNS in data_loader.py."
        )

    return df


def get_data_shape_summary(df: pd.DataFrame) -> dict:
    """
    Return a simple dictionary summarising the loaded dataset's dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Any loaded DataFrame.

    Returns
    -------
    dict
        Keys: n_rows, n_columns, n_cells
    """
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "n_cells": df.shape[0] * df.shape[1],
    }
