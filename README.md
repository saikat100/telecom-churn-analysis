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
Status
🚧 Class 1 of 3 — Project Foundation, Data Ingestion and Profiling
