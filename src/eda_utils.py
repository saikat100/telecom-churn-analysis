# src/eda_utils.py
"""
eda_utils.py
------------
Reusable plotting and aggregation utilities for the telecom churn EDA.
All plot functions follow a consistent interface:
 - Accept a DataFrame and axis/figure parameters
 - Return the matplotlib Figure object (for saving or display)
 - Accept a title override for flexibility
 - Never call plt.show() internally — caller decides when to render
All aggregation functions return a DataFrame, not a plot,
so results can be saved, tested, or displayed independently.
Analytical questions served:
 Q1 — who_churn_by_segment()
 Q2 — churn_by_tenure_cohort()
 Q3 — multivariate_churn_heatmap(), behavioral_signal_summary()
 Q4 — risk_concentration_table()
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional, List, Tuple

# ── Style Constants ────────────────────────────────────────────────────────────
CHURN_PALETTE = {"No": "#2ecc71", "Yes": "#e74c3c"}
COHORT_PALETTE = "Blues_d"
FIGURE_DPI = 120
BASE_FONT_SIZE = 11

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": FIGURE_DPI,
})

# ── Aggregation Functions ──────────────────────────────────────────────────────
def churn_rate_by_segment(df: pd.DataFrame,
                          segment_col: str,
                          target_col: str = "churn",
                          churn_value: str = "Yes") -> pd.DataFrame:
    """
    Compute churn rate, customer count, and churned count
    for each unique value of segment_col.
    Parameters
    ----------
    df : pd.DataFrame
    segment_col : str
        Categorical column to segment by.
    target_col : str
        Binary target column. Default: 'churn'.
    churn_value : str
        Value in target_col that represents a churn event. Default: 'Yes'.
    Returns
    -------
    pd.DataFrame
        Columns: segment, total_customers, churned, churn_rate_pct.
        Sorted by churn_rate_pct descending.
    """
    grouped = df.groupby(segment_col, observed=True).agg(
        total_customers=(target_col, "count"),
        churned=(target_col, lambda x: (x == churn_value).sum()),
    ).reset_index()
    grouped["churn_rate_pct"] = (
        grouped["churned"] / grouped["total_customers"] * 100
    ).round(2)
    grouped = grouped.rename(columns={segment_col: "segment"})
    return grouped.sort_values("churn_rate_pct", ascending=False).reset_index(drop=True)


def behavioral_signal_summary(df: pd.DataFrame,
                               feature_cols: List[str],
                               target_col: str = "churn",
                               churn_value: str = "Yes") -> pd.DataFrame:
    """
    For each feature in feature_cols, compute the mean value
    separately for churned and non-churned customers, and the
    difference between them.
    Used for numeric/binary features — tells you whether churned
    customers score higher or lower on each behavioral signal.
    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : List[str]
        Numeric or binary columns to compare.
    target_col : str
    churn_value : str
    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_churned, mean_retained, difference, direction.
        Sorted by absolute difference descending.
    """
    records = []
    for col in feature_cols:
        mean_churned = df[df[target_col] == churn_value][col].mean()
        mean_retained = df[df[target_col] != churn_value][col].mean()
        diff = mean_churned - mean_retained
        direction = "higher in churned" if diff > 0 else "lower in churned"
        records.append({
            "feature": col,
            "mean_churned": round(float(mean_churned), 3),
            "mean_retained": round(float(mean_retained), 3),
            "difference": round(float(diff), 3),
            "direction": direction,
        })
    result = pd.DataFrame(records)
    result["abs_difference"] = result["difference"].abs()
    return result.sort_values("abs_difference", ascending=False).drop(
        columns="abs_difference"
    ).reset_index(drop=True)


def churn_by_tenure_cohort(df: pd.DataFrame,
                            cohort_col: str = "tenure_group",
                            target_col: str = "churn",
                            churn_value: str = "Yes") -> pd.DataFrame:
    """
    Compute churn rate and key metrics per tenure cohort.
    Used for cohort analysis answering Q2.
    Parameters
    ----------
    df : pd.DataFrame
    cohort_col : str
        Ordered categorical tenure group column.
    target_col : str
    churn_value : str
    Returns
    -------
    pd.DataFrame
        Columns: cohort, total_customers, churned, churn_rate_pct,
        avg_monthly_charges, avg_addon_services.
    """
    result = df.groupby(cohort_col, observed=True).agg(
        total_customers=(target_col, "count"),
        churned=(target_col, lambda x: (x == churn_value).sum()),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_addon_services=("total_addon_services", "mean"),
    ).reset_index()
    result["churn_rate_pct"] = (
        result["churned"] / result["total_customers"] * 100
    ).round(2)
    result["avg_monthly_charges"] = result["avg_monthly_charges"].round(2)
    result["avg_addon_services"] = result["avg_addon_services"].round(2)
    result = result.rename(columns={cohort_col: "cohort"})
    return result


def multivariate_churn_pivot(df: pd.DataFrame,
                              row_col: str,
                              col_col: str,
                              target_col: str = "churn",
                              churn_value: str = "Yes") -> pd.DataFrame:
    """
    Build a pivot table of churn rates for two categorical features simultaneously.
    Used for multivariate/interaction analysis answering Q3.
    Parameters
    ----------
    df : pd.DataFrame
    row_col : str
        Categorical column for pivot rows.
    col_col : str
        Categorical column for pivot columns.
    target_col : str
    churn_value : str
    Returns
    -------
    pd.DataFrame
        Pivot table where each cell is the churn rate (%) for
        the intersection of row and column values.
    """
    df = df.copy()
    df["_churned"] = (df[target_col] == churn_value).astype(int)
    pivot = df.pivot_table(
        values="_churned",
        index=row_col,
        columns=col_col,
        aggfunc="mean",
    ) * 100
    return pivot.round(2)


def risk_concentration_table(df: pd.DataFrame,
                              segment_cols: List[str],
                              target_col: str = "churn",
                              churn_value: str = "Yes") -> pd.DataFrame:
    """
    Build a multi-column segmentation table showing what percentage
    of total churned customers each combined segment accounts for.
    Answers Q4: how concentrated is churn risk?
    Parameters
    ----------
    df : pd.DataFrame
    segment_cols : List[str]
        List of categorical columns to group by simultaneously.
    target_col : str
    churn_value : str
    Returns
    -------
    pd.DataFrame
        Sorted by churned_count descending, with cumulative share column.
    """
    total_churned = (df[target_col] == churn_value).sum()
    grouped = df.groupby(segment_cols, observed=True).agg(
        total_customers=(target_col, "count"),
        churned_count=(target_col, lambda x: (x == churn_value).sum()),
    ).reset_index()
    grouped["churn_rate_pct"] = (
        grouped["churned_count"] / grouped["total_customers"] * 100
    ).round(2)
    grouped["share_of_total_churn_pct"] = (
        grouped["churned_count"] / total_churned * 100
    ).round(2)
    grouped = grouped.sort_values("churned_count", ascending=False).reset_index(drop=True)
    grouped["cumulative_churn_share_pct"] = (
        grouped["share_of_total_churn_pct"].cumsum().round(2)
    )
    return grouped


# ── Plot Functions ─────────────────────────────────────────────────────────────
def plot_churn_rate_by_segment(segment_df: pd.DataFrame,
                                title: str,
                                figsize: Tuple = (10, 5)) -> plt.Figure:
    """
    Horizontal bar chart of churn rate per segment value.
    Color encodes churn rate severity (low = green, high = red).
    Parameters
    ----------
    segment_df : pd.DataFrame
        Output of churn_rate_by_segment().
    title : str
    figsize : Tuple
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    max_rate = segment_df["churn_rate_pct"].max()
    colors = [
        "#e74c3c" if r > max_rate * 0.7
        else "#f39c12" if r > max_rate * 0.4
        else "#2ecc71"
        for r in segment_df["churn_rate_pct"]
    ]
    bars = ax.barh(
        segment_df["segment"].astype(str),
        segment_df["churn_rate_pct"],
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    # Annotate each bar with rate and count
    for bar, (_, row) in zip(bars, segment_df.iterrows()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['churn_rate_pct']:.1f}% (n={row['total_customers']:,})",
            va="center", fontsize=9,
        )
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_cohort_churn_trend(cohort_df: pd.DataFrame,
                             title: str = "Churn Rate Across Customer Lifecycle Cohorts",
                             figsize: Tuple = (11, 5)) -> plt.Figure:
    """
    Line chart showing churn rate across ordered tenure cohorts,
    with customer count annotated at each point.
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Output of churn_by_tenure_cohort().
    title : str
    figsize : Tuple
    Returns
    -------
    plt.Figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    # Primary axis: churn rate line
    ax1.plot(
        cohort_df["cohort"].astype(str),
        cohort_df["churn_rate_pct"],
        marker="o", linewidth=2.5, color="#e74c3c", markersize=8, label="Churn Rate (%)"
    )
    for _, row in cohort_df.iterrows():
        ax1.annotate(
            f"{row['churn_rate_pct']:.1f}%\n(n={row['total_customers']:,})",
            xy=(str(row["cohort"]), row["churn_rate_pct"]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9,
        )
    # Secondary axis: avg add-on services bar
    ax2 = ax1.twinx()
    ax2.bar(
        cohort_df["cohort"].astype(str),
        cohort_df["avg_addon_services"],
        alpha=0.25, color="steelblue", label="Avg Add-on Services"
    )
    ax2.set_ylabel("Avg Add-on Services", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylabel("Churn Rate (%)", color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_xlabel("Customer Lifecycle Cohort")
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    return fig


def plot_multivariate_heatmap(pivot_df: pd.DataFrame,
                               title: str,
                               figsize: Tuple = (10, 5)) -> plt.Figure:
    """
    Heatmap of churn rates for a two-way feature interaction.
    Cell values are churn rate percentages.
    Parameters
    ----------
    pivot_df : pd.DataFrame
        Output of multivariate_churn_pivot().
    title : str
    figsize : Tuple
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Churn Rate (%)"},
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_behavioral_signals(signal_df: pd.DataFrame,
                             title: str = "Behavioral Signal Comparison: Churned vs Retained",
                             figsize: Tuple = (10, 5)) -> plt.Figure:
    """
    Diverging bar chart comparing mean feature values between
    churned and retained customers.
    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of behavioral_signal_summary().
    title : str
    figsize : Tuple
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in signal_df["difference"]]
    ax.barh(
        signal_df["feature"],
        signal_df["difference"],
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Difference in Mean Value (Churned − Retained)")
    ax.set_title(title)
    ax.invert_yaxis()
    for i, (_, row) in enumerate(signal_df.iterrows()):
        offset = 0.002 if row["difference"] >= 0 else -0.002
        ax.text(
            row["difference"] + offset, i,
            f" {row['direction']}", va="center", fontsize=8,
        )
    plt.tight_layout()
    return fig


def plot_distribution_by_churn(df: pd.DataFrame,
                                numeric_col: str,
                                target_col: str = "churn",
                                title: Optional[str] = None,
                                figsize: Tuple = (10, 5)) -> plt.Figure:
    """
    KDE plot comparing the distribution of a numeric column
    for churned vs retained customers.
    Parameters
    ----------
    df : pd.DataFrame
    numeric_col : str
    target_col : str
    title : Optional[str]
    figsize : Tuple
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    for churn_val, color in CHURN_PALETTE.items():
        subset = df[df[target_col] == churn_val][numeric_col]
        subset.plot.kde(ax=ax, label=f"Churn = {churn_val}",
                        color=color, linewidth=2)
        ax.axvline(subset.median(), color=color, linestyle="--",
                   linewidth=1, alpha=0.7)
    ax.set_xlabel(numeric_col)
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution of {numeric_col} by Churn Status")
    ax.legend()
    plt.tight_layout()
    return fig