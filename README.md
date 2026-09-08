# Telecom Customer Churn Analysis

## Analytical Questions This Project Answers
1. Who is churning — which segments show the highest churn rate?
2. When do customers tend to churn — what are the critical tenure windows?
3. What behavioral signals precede churn — service usage, billing, add-on patterns?
4. How severe is the risk concentration — can we produce an actionable target list?

## Project Structure
- `data/raw/` — Original immutable dataset (never edited)
- `data/processed/` — Cleaned and feature-engineered outputs
- `src/` — All reusable Python modules (logic lives here, not in notebooks)
- `notebooks/` — Thin analysis notebooks that import from src/
- `tests/` — pytest unit tests for src/ modules
- `reports/` — Programmatically generated markdown and figure outputs
- `docs/` — Data dictionary and project documentation
- `app/` — Streamlit dashboard (Class 3)

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

Run the notebooks in order — each one writes the inputs the next one reads:

```bash
jupyter lab
```

1. `notebooks/01_data_ingestion_and_profiling.ipynb` — profiles the raw export
   and regenerates `docs/data_dictionary.md` and `reports/data_profile.md`
2. `notebooks/02_cleaning_and_feature_engineering.ipynb` — cleans, treats
   outliers, derives features, and writes
   `data/processed/telecom_customers_processed.csv`
3. `notebooks/03_exploratory_data_analysis.ipynb` — answers the four questions
   and writes `reports/eda_summary.md` plus the figures

Then launch the dashboard, which reads the processed dataset:

```bash
streamlit run app/dashboard.py
```

Run the test suite with:

```bash
pytest -q
```

## Pipeline Stages

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | Load the raw CSV and validate its structure |
| `src/profiler.py` | Dtype, missing, duplicate, cardinality and target reports |
| `src/cleaner.py` | Type fixes, canonical categories, imputation, deduplication |
| `src/outlier_handler.py` | Per-column outlier strategy with an audit trail |
| `src/feature_engineer.py` | Six derived features, each with a stated hypothesis |
| `src/eda_utils.py` | Segment aggregations and the plots built from them |

## Known Data Characteristics

- The raw export spells some categories inconsistently (`male` / `M` / `Male`,
  `month-to-month` / `Month-to-month`). `standardise_categorical_values` folds
  these together; without it, segment churn rates are computed on split groups.
- 64 customers have a blank `total_charges`. All of them have zero tenure, so
  this is a structural zero rather than missing data, and is imputed as 0.0.
- `payment_method` values are `Bank transfer` / `Credit card`, without the
  `(automatic)` suffix used in the original public version of this dataset.

## Status
Complete — ingestion, profiling, cleaning, outlier treatment, feature
engineering, exploratory analysis, and the dashboard.
