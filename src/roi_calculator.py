"""
roi_calculator.py — Campaign ROI, CPA, lift, and sensitivity analysis.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
 
# ─── Default Assumptions (override as needed) ────────────────────────────────
 
DEFAULT_COST_PER_CONTACT = 50       # ₹ per outbound contact
DEFAULT_REVENUE_PER_CONVERSION = 8000  # ₹ average deposit / product revenue
BASELINE_CONVERSION_RATE = 0.05     # assumed pre-campaign baseline (5%)
 
 
# ─── Core Metrics ────────────────────────────────────────────────────────────
 
def compute_campaign_roi(
    n_contacts: int,
    n_conversions: int,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
    revenue_per_conversion: float = DEFAULT_REVENUE_PER_CONVERSION,
) -> dict:
    """
    Calculate overall campaign ROI.
 
    ROI = (Revenue - Cost) / Cost × 100
    """
    total_cost = n_contacts * cost_per_contact
    total_revenue = n_conversions * revenue_per_conversion
    profit = total_revenue - total_cost
    roi = (profit / total_cost) * 100 if total_cost > 0 else 0.0
    cpa = total_cost / n_conversions if n_conversions > 0 else float("inf")
    conversion_rate = n_conversions / n_contacts if n_contacts > 0 else 0.0
 
    return {
        "n_contacts": n_contacts,
        "n_conversions": n_conversions,
        "conversion_rate_pct": round(conversion_rate * 100, 2),
        "total_cost": round(total_cost, 2),
        "total_revenue": round(total_revenue, 2),
        "profit": round(profit, 2),
        "roi_pct": round(roi, 2),
        "cpa": round(cpa, 2),
    }
 
 
def compute_channel_roi(
    df: pd.DataFrame,
    channel_col: str = "contact",
    target_col: str = "y",
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
    revenue_per_conversion: float = DEFAULT_REVENUE_PER_CONVERSION,
) -> pd.DataFrame:
    """
    Calculate ROI metrics broken down by channel (cellular vs telephone).
    """
    rows = []
    for channel, grp in df.groupby(channel_col):
        n_contacts = len(grp)
        n_conv = grp[target_col].sum()
        metrics = compute_campaign_roi(
            n_contacts, n_conv, cost_per_contact, revenue_per_conversion
        )
        metrics["channel"] = channel
        rows.append(metrics)
    result = pd.DataFrame(rows).set_index("channel")
    return result
 
 
def compute_segment_roi(
    df: pd.DataFrame,
    segment_col: str = "segment_label",
    target_col: str = "y",
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
    revenue_per_conversion: float = DEFAULT_REVENUE_PER_CONVERSION,
) -> pd.DataFrame:
    """
    Calculate ROI metrics broken down by customer segment.
    """
    rows = []
    for seg, grp in df.groupby(segment_col):
        n_contacts = len(grp)
        n_conv = grp[target_col].sum()
        metrics = compute_campaign_roi(
            n_contacts, n_conv, cost_per_contact, revenue_per_conversion
        )
        metrics["segment"] = seg
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("segment")
 
 
# ─── Incremental Lift Analysis ───────────────────────────────────────────────
 
def compute_incremental_lift(
    n_contacts: int,
    n_conversions: int,
    baseline_rate: float = BASELINE_CONVERSION_RATE,
    revenue_per_conversion: float = DEFAULT_REVENUE_PER_CONVERSION,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
) -> dict:
    """
    Measure how much revenue is attributable to the campaign vs the baseline.
 
    Incremental conversions = Actual conversions − Expected without campaign
    Incremental revenue = Incremental conversions × Revenue per conversion
    """
    observed_rate = n_conversions / n_contacts if n_contacts > 0 else 0
    baseline_conversions = n_contacts * baseline_rate
    incremental_conversions = n_conversions - baseline_conversions
    incremental_revenue = incremental_conversions * revenue_per_conversion
    total_cost = n_contacts * cost_per_contact
    incremental_roi = (
        (incremental_revenue - total_cost) / total_cost * 100
        if total_cost > 0 else 0
    )
    lift_pct = (
        (observed_rate - baseline_rate) / baseline_rate * 100
        if baseline_rate > 0 else 0
    )
 
    return {
        "observed_conversion_rate_pct": round(observed_rate * 100, 2),
        "baseline_conversion_rate_pct": round(baseline_rate * 100, 2),
        "lift_pct": round(lift_pct, 2),
        "baseline_conversions": round(baseline_conversions, 0),
        "incremental_conversions": round(incremental_conversions, 0),
        "incremental_revenue": round(incremental_revenue, 2),
        "total_cost": round(total_cost, 2),
        "incremental_roi_pct": round(incremental_roi, 2),
    }
 
 
# ─── Sensitivity Analysis ────────────────────────────────────────────────────
 
def sensitivity_analysis(
    n_contacts: int,
    n_conversions: int,
    conversion_rate_deltas: list = None,
    cost_deltas: list = None,
    revenue_per_conversion: float = DEFAULT_REVENUE_PER_CONVERSION,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
) -> pd.DataFrame:
    """
    Show how ROI changes as conversion rate and cost-per-contact vary.
    """
    if conversion_rate_deltas is None:
        conversion_rate_deltas = [-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30]
    if cost_deltas is None:
        cost_deltas = [-0.20, -0.10, 0, 0.10, 0.20]
 
    base_rate = n_conversions / n_contacts if n_contacts > 0 else 0
    rows = []
    for cr_delta in conversion_rate_deltas:
        new_rate = base_rate * (1 + cr_delta)
        new_conv = int(n_contacts * new_rate)
        for cost_delta in cost_deltas:
            new_cost = cost_per_contact * (1 + cost_delta)
            metrics = compute_campaign_roi(
                n_contacts, new_conv, new_cost, revenue_per_conversion
            )
            rows.append({
                "conversion_rate_change_pct": round(cr_delta * 100, 0),
                "cost_change_pct": round(cost_delta * 100, 0),
                "roi_pct": metrics["roi_pct"],
                "cpa": metrics["cpa"],
                "profit": metrics["profit"],
            })
    return pd.DataFrame(rows)
 
 
# ─── Visualisations ──────────────────────────────────────────────────────────
 
def plot_channel_roi(channel_df: pd.DataFrame, save_path: str = None):
    metrics = ["conversion_rate_pct", "cpa", "roi_pct"]
    titles = ["Conversion Rate (%)", "Cost per Acquisition (₹)", "ROI (%)"]
    colors = ["#4C72B0", "#DD8452"]
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(axes, metrics, titles):
        channel_df[metric].plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.set_ylabel(title)
        for bar in ax.patches:
            ax.annotate(
                f"{bar.get_height():,.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9
            )
 
    plt.suptitle("Channel ROI Comparison: Cellular vs Telephone", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_segment_roi(segment_df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("Set2", len(segment_df))
 
    segment_df["roi_pct"].plot(kind="barh", ax=axes[0], color=palette)
    axes[0].set_title("ROI (%) by Customer Segment")
    axes[0].set_xlabel("ROI (%)")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
 
    segment_df["cpa"].plot(kind="barh", ax=axes[1], color=palette)
    axes[1].set_title("Cost per Acquisition (₹) by Segment")
    axes[1].set_xlabel("CPA (₹)")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_sensitivity_heatmap(sensitivity_df: pd.DataFrame, metric: str = "roi_pct", save_path: str = None):
    pivot = sensitivity_df.pivot(
        index="conversion_rate_change_pct",
        columns="cost_change_pct",
        values=metric
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, linewidths=0.4, cbar_kws={"label": metric}
    )
    plt.title(f"Sensitivity Analysis — {metric} vs Conversion Rate & Cost Changes")
    plt.xlabel("Cost per Contact Change (%)")
    plt.ylabel("Conversion Rate Change (%)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_roi_waterfall(metrics: dict, save_path: str = None):
    """Waterfall chart: Revenue build-up."""
    labels = ["Total Revenue", "− Contact Costs", "= Net Profit"]
    values = [metrics["total_revenue"], -metrics["total_cost"], metrics["profit"]]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
 
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, [abs(v) for v in values], color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(abs(v) for v in values) * 0.01,
            f"₹{val:,.0f}", ha="center", va="bottom", fontsize=10
        )
    plt.title(f"Campaign P&L — ROI: {metrics['roi_pct']}%")
    plt.ylabel("Amount (₹)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def print_roi_summary(metrics: dict):
    print("=" * 45)
    print("       CAMPAIGN ROI SUMMARY")
    print("=" * 45)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>10,.2f}")
        else:
            print(f"  {k:<35} {v:>10,}")
    print("=" * 45)