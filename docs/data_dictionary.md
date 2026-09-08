# Data Dictionary — Telecom Customer Churn Dataset

**Source:** `data/raw/telecom_customers.csv` 
**Rows:** 5,001 | **Columns:** 21 

---

| Column            | Stored Type   |   Unique Values |   Missing (NaN) | Description                                                                                                                                   |
|:------------------|:--------------|----------------:|----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------|
| customer_id       | object        |            5000 |               0 | Unique identifier per customer. Expected: no duplicates.                                                                                      |
| gender            | object        |               5 |               0 | Customer gender. Values: Male / Female.                                                                                                       |
| senior_citizen    | int64         |               2 |               0 | Whether customer is a senior citizen. Values: 0 (No), 1 (Yes). Note: stored as int — will be standardized in cleaning.                        |
| partner           | object        |               2 |               0 | Whether the customer has a partner. Values: Yes / No.                                                                                         |
| dependents        | object        |               2 |             200 | Whether the customer has dependents. Values: Yes / No.                                                                                        |
| tenure_months     | int64         |              74 |               0 | Number of months the customer has been with the company. Range: 0–72.                                                                         |
| phone_service     | object        |               2 |               0 | Whether the customer has a phone service. Values: Yes / No.                                                                                   |
| multiple_lines    | object        |               3 |               0 | Whether the customer has multiple phone lines. Values: Yes / No / No phone service.                                                           |
| internet_service  | object        |               3 |               0 | Type of internet service. Values: DSL / Fiber optic / No.                                                                                     |
| online_security   | object        |               3 |               0 | Whether the customer has online security add-on. Values: Yes / No / No internet service.                                                      |
| online_backup     | object        |               3 |               0 | Whether the customer has online backup add-on. Values: Yes / No / No internet service.                                                        |
| device_protection | object        |               3 |               0 | Whether the customer has device protection add-on. Values: Yes / No / No internet service.                                                    |
| tech_support      | object        |               3 |               0 | Whether the customer has tech support add-on. Values: Yes / No / No internet service.                                                         |
| streaming_tv      | object        |               3 |               0 | Whether the customer has streaming TV service. Values: Yes / No / No internet service.                                                        |
| streaming_movies  | object        |               3 |               0 | Whether the customer has streaming movies service. Values: Yes / No / No internet service.                                                    |
| contract_type     | object        |               4 |               0 | Type of contract. Values: Month-to-month / One year / Two year.                                                                               |
| paperless_billing | object        |               2 |               0 | Whether the customer uses paperless billing. Values: Yes / No.                                                                                |
| payment_method    | object        |               4 |             150 | Payment method used. Values: Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic).                           |
| monthly_charges   | float64       |            3414 |               0 | Amount charged to the customer each month. Type: float.                                                                                       |
| total_charges     | object        |            4913 |               0 | Total amount charged over the customer's tenure. Type: float. Known issue: stored as object in source — requires type conversion in cleaning. |
| churn             | object        |               2 |               0 | TARGET VARIABLE. Whether the customer churned. Values: Yes / No.                                                                              |

---
*Generated programmatically. Do not edit manually.*
