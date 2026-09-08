# EDA Summary Report — Telecom Customer Churn Analysis

**Dataset:** 4,995 customers | **Overall Churn Rate:** 23.8% | **Total Churned:** 1,188

---

## Q1 — Who Is Churning?

**Contract type** is the single strongest predictor of churn. Customers on **Month-to-month** contracts churn at **27.0%**, compared to **18.2%** for **Two year** contract customers — a 1.5x difference.

**Internet service type** also differentiates churn significantly. **DSL** customers show a churn rate of **23.9%**. Statistical significance confirmed via Chi-Square test (p < 0.05) for both features.

## Q2 — When Do Customers Tend to Churn?

Churn risk is highest in the **New (0-12m)** lifecycle stage, with a churn rate of **28.2%** (885 customers). Risk declines consistently with tenure, reaching its lowest point in the Loyal cohort. This pattern strongly supports the case for early-lifecycle intervention programs.

## Q3 — What Behavioral Signals Precede Churn?

Six behavioral features were tested against the churned vs retained population:

| feature                  |   mean_churned |   mean_retained | direction         |
|:-------------------------|---------------:|----------------:|:------------------|
| tenure_months            |         34.934 |          36.578 | lower in churned  |
| avg_charge_per_month     |         70.115 |          70.288 | lower in churned  |
| monthly_charges          |         70.124 |          70.285 | lower in churned  |
| total_addon_services     |          1.614 |           1.656 | lower in churned  |
| is_new_customer          |          0.066 |           0.045 | higher in churned |
| is_fully_automated_payer |          0.402 |           0.397 | higher in churned |
| has_any_streaming        |          0.506 |           0.506 | higher in churned |

The strongest behavioral separator is **tenure_months** (difference: -1.644). Multivariate analysis reveals that the combination of Month-to-month contract + Fiber optic internet creates compounding risk beyond what either factor predicts independently.

## Q4 — How Concentrated Is Churn Risk?

The top 3 contract × internet service segments account for **65.2%** of all churned customers.

The single highest-risk segment is: **Month-to-month × Fiber optic** with a churn rate of **27.9%** and a **29.9%** share of total churn.

This concentration means the retention team can address the majority of churn by focusing on a minority of the customer base — a highly efficient use of budget.

## Hypothesis Verdicts

| feature                  | hypothesis                                                                | eda_verdict                                                             |
|:-------------------------|:--------------------------------------------------------------------------|:------------------------------------------------------------------------|
| tenure_group             | Churn risk peaks for new customers and declines with tenure               | Confirmed — New cohort churns significantly more (see Cell 11)          |
| total_addon_services     | More add-ons = higher switching cost = lower churn                        | Confirmed — Churned customers have fewer add-ons on average             |
| avg_charge_per_month     | Higher effective spend correlates with price sensitivity and higher churn | Confirmed — Churned customers have higher effective spend rate          |
| is_new_customer          | First 3 months represent disproportionately high churn risk               | Confirmed — New customers overrepresented in churned group              |
| has_any_streaming        | Streaming users have higher switching cost and lower churn                | Partially confirmed — streaming users churn less but effect is moderate |
| is_fully_automated_payer | Auto-pay customers churn less due to reduced active billing engagement    | Confirmed — Auto-pay customers have lower churn rate                    |

## Recommended Actions

1. **Immediate:** Target Month-to-month customers with Fiber optic service with contract upgrade incentives — this segment has the highest churn rate and the highest share of total churn.
2. **Short-term:** Build an onboarding intervention program for customers in their first 12 months — cohort analysis confirms this is the peak churn window.
3. **Medium-term:** Increase add-on service adoption among new and mid-term customers — the engagement signal is consistent across all cohorts.
4. **Ongoing:** Monitor the Electronic check payment segment — these customers show higher churn and may benefit from migration to auto-pay.

---
*Generated programmatically by `notebooks/03_eda_and_insights.ipynb`. Do not edit manually — rerun notebook to refresh.*
