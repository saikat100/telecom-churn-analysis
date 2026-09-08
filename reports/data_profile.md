# Data Profile Report — Telecom Customer Churn Dataset
**Rows:** 5,001 | **Columns:** 21

---
## 1. Dtype Audit
**1 column(s) flagged as dtype-suspicious:**
- `total_charges`

## 2. Missing Value Summary
| column         |   nan_count |   blank_string_count |   combined_missing |   missing_pct |
|:---------------|------------:|---------------------:|-------------------:|--------------:|
| dependents     |         200 |                    0 |                200 |          4    |
| payment_method |         150 |                    0 |                150 |          3    |
| total_charges  |           0 |                   64 |                 64 |          1.28 |

## 3. Duplicate Summary
- Full row duplicates: **1**
- ID-level duplicates (`customer_id`): **1**

## 4. Target Variable Distribution
| class   |   count |   percentage | imbalance_flag    |
|:--------|--------:|-------------:|:------------------|
| No      |    3813 |        76.24 |                   |
| Yes     |    1188 |        23.76 | ⚠  minority class |

---
*Generated programmatically by `src/profiler.py`. Do not edit manually — rerun the notebook to refresh.*
