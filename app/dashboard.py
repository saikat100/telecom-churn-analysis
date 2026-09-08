import streamlit as st
import os
import pandas as pd
import plotly.express as px

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "processed", "telecom_customers_processed.csv"
)
HIGH_RISK_CONTRACT = "Month-to-month"
HIGH_RISK_INTERNET = "Fiber optic"
HIGH_RISK_TENURE_MONTHS = 12

CHURN_COLOR_MAP = {
    "Yes": "#e74c3c",
    "No": "#2ecc71"
}

# Step 1: Page Setup
st.set_page_config(
    page_title="Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Telecom Customer Churn Dashboard")
st.write(
    "This dashboard helps us answer one big question: "
    "**Which customers are most likely to leave (churn), and why?**"
)

# Step 2: Load the data
if not os.path.exists(DATA_PATH):
    st.error(f"Could not find data at {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

required_cols = {
    "contract_type", "internet_service", "churn",
    "tenure_months", "monthly_charges", "total_addon_services",
    "customer_id", "gender", "payment_method", "senior_citizen",
}

missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Data file is missing required columns: {sorted(missing_cols)}")
    st.stop()

st.divider()

# Step 3: sidebar filters
contract_choice = st.sidebar.multiselect(
    "Contract Type",
    options=sorted(df['contract_type'].unique()),
    default=sorted(df['contract_type'].unique()),
)

internet_choice = st.sidebar.multiselect(
    "Internet Service",
    options=sorted(df["internet_service"].unique()),
    default=sorted(df["internet_service"].unique()),
)

contract_mask = df["contract_type"].isin(contract_choice)
internet_mask = df["internet_service"].isin(internet_choice)

filtered_df = df[contract_mask & internet_mask]

st.sidebar.write(f"Showing **{len(filtered_df):,}** of **{len(df):,}** customers")

# Step 4: Key Performance Index (KPIs)
total_customers = len(filtered_df)
total_churned = int((filtered_df["churn"] == "Yes").sum())
churn_rate = (total_churned / total_customers) * 100
avg_monthly_charge = filtered_df["monthly_charges"].mean()

st.header("Key Performance Index")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Customers Who Left", f"{total_churned:,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Avg Monthly Charge", f"${avg_monthly_charge:.2f}")

st.divider()

# Step 5: Overall churn pie chart
st.header("Churn Split")
st.write("Out of all customers we're looking at, how many stayed vs left?")

if filtered_df.empty:
    st.warning("No customers left.")
else:
    churn_counts = filtered_df['churn'].value_counts().reset_index()
    print(churn_counts)
    print(churn_counts.columns)

fig_pie = px.pie(
    churn_counts,
    names="churn",
    values="count",
    color="churn",
    color_discrete_map=CHURN_COLOR_MAP,
    title="Customers: Churned vs Retained",
    hole=0.4,
)

st.plotly_chart(
    fig_pie,
    use_container_width=True,
)

st.info("Findings: ...")
st.info("Action: ...")

st.divider()

st.header("Who Is Churning? — By Contract Type")
st.write("Let's compare churn rate across different contract types.")

if filtered_df.empty:
    st.warning("No data matches the current filters.")
else:
    contract_summary = (
        filtered_df.groupby("contract_type")["churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="churn_rate")
        .sort_values("churn_rate")
    )

    fig_contract = px.bar(
        contract_summary,
        x="churn_rate",
        y="contract_type",
        orientation="h",
        text="churn_rate",
        title="Churn Rate (%) by Contract Type",
        labels={"churn_rate": "Churn Rate (%)", "contract_type": "Contract Type"},
    )
    fig_contract.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_contract, use_container_width=True)

    st.info(
        "💡 Teaching point: Month-to-month customers usually churn the most "
        "because they have no commitment keeping them in place."
    )

st.divider()

st.header("How Many Customers Are on Each Contract?")
st.write("Before judging churn rate, it helps to see group sizes.")

if filtered_df.empty:
    st.warning("No data matches the current filters.")
else:
    contract_counts = filtered_df["contract_type"].value_counts().reset_index()
    contract_counts.columns = ["contract_type", "count"]

    fig_counts = px.bar(
        contract_counts,
        x="contract_type",
        y="count",
        text="count",
        title="Number of Customers by Contract Type",
        labels={"contract_type": "Contract Type", "count": "Number of Customers"},
    )
    fig_counts.update_traces(textposition="outside")
    st.plotly_chart(fig_counts, use_container_width=True)

    st.info(
        "💡 Teaching point: A high churn *rate* on a small group matters less "
        "than the same rate on a huge group — always check the count too."
    )

st.divider()

# =============================================================================
# STEP 8: WHEN DO CUSTOMERS CHURN? (TENURE HISTOGRAM)
# =============================================================================
st.header("When Do Customers Churn?")
st.write("Does churn happen early on, or later? Let's check tenure (months as a customer).")

if filtered_df.empty:
    st.warning("No data matches the current filters.")
else:
    fig_tenure = px.histogram(
        filtered_df,
        x="tenure_months",
        color="churn",
        color_discrete_map=CHURN_COLOR_MAP,
        barmode="overlay",
        nbins=30,
        title="Customer Tenure Distribution: Churned vs Retained",
        labels={"tenure_months": "Months as a Customer"},
    )
    st.plotly_chart(fig_tenure, use_container_width=True)

    st.info(
        "💡 Teaching point: Look for a tall red bar near month 0 — "
        "that tells us new customers are the most likely to leave early."
    )

st.divider()

st.header("")
st.write("...")

high_risk_df = filtered_df[
    (filtered_df["contract_type"] == HIGH_RISK_CONTRACT)
    & (filtered_df["internet_service"] == HIGH_RISK_INTERNET)
    & (filtered_df["tenure_months"] <= HIGH_RISK_TENURE_MONTHS)
][["customer_id", "tenure_months", "monthly_charges", "churn"]]

st.dataframe(high_risk_df, use_container_width=True)
st.caption("....")

st.download_button(
    "⬇ Download High-Risk Customer List",
    data=high_risk_df.to_csv(index=False),
    file_name='high_risk_customer_list.csv',
    mime='text/csv',
    disabled=high_risk_df.empty
)

st.divider()
st.caption("Telecom Customer Churn Dashboard")