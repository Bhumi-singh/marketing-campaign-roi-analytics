"""
roi_calculator.py
Campaign ROI measurement: CPA, revenue, lift, sensitivity analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from typing import Dict

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Default assumptions (₹ Indian Rupees) ─────────────────────────────────────
DEFAULT_ASSUMPTIONS = {
    "cost_per_contact_cellular"  : 45,     # ₹ per cellular call
    "cost_per_contact_telephone" : 25,     # ₹ per telephone call
    "revenue_per_conversion"     : 8_500,  # ₹ average term deposit value (annualised)
    "baseline_conversion_rate"   : 0.05,   # 5% organic baseline
}


def compute_channel_roi(df: pd.DataFrame, assumptions: Dict = None) -> pd.DataFrame:
    if assumptions is None:
        assumptions = DEFAULT_ASSUMPTIONS

    results = []
    for channel in df["contact"].unique():
        sub = df[df["contact"] == channel]
        n           = len(sub)
        conversions = sub["y_binary"].sum()
        conv_rate   = conversions / n

        cost_per    = assumptions.get(f"cost_per_contact_{channel}", 35)
        total_cost  = n * cost_per
        revenue     = conversions * assumptions["revenue_per_conversion"]
        profit      = revenue - total_cost
        roi         = (revenue - total_cost) / total_cost * 100 if total_cost > 0 else 0
        cpa         = total_cost / conversions if conversions > 0 else np.nan

        results.append({
            "channel"         : channel,
            "total_contacts"  : n,
            "conversions"     : conversions,
            "conversion_rate" : round(conv_rate, 4),
            "cost_per_contact": cost_per,
            "total_cost_inr"  : total_cost,
            "revenue_inr"     : revenue,
            "profit_inr"      : profit,
            "roi_pct"         : round(roi, 2),
            "cpa_inr"         : round(cpa, 2),
        })

    result_df = pd.DataFrame(results).set_index("channel")
    print("\n── Channel ROI ──────────────────────────────────────")
    print(result_df[["total_contacts","conversions","conversion_rate",
                      "roi_pct","cpa_inr"]].to_string())
    return result_df


def compute_segment_roi(df: pd.DataFrame, assumptions: Dict = None) -> pd.DataFrame:
    if assumptions is None:
        assumptions = DEFAULT_ASSUMPTIONS
    if "segment_label" not in df.columns:
        print("[warn] No segment_label column — skipping segment ROI")
        return pd.DataFrame()

    # Assume cellular cost for all (simplification)
    cost_per = assumptions["cost_per_contact_cellular"]
    rows = []
    for seg, sub in df.groupby("segment_label"):
        n           = len(sub)
        conversions = sub["y_binary"].sum()
        conv_rate   = conversions / n
        total_cost  = n * cost_per
        revenue     = conversions * assumptions["revenue_per_conversion"]
        roi         = (revenue - total_cost) / total_cost * 100 if total_cost > 0 else 0
        cpa         = total_cost / conversions if conversions > 0 else np.nan
        rows.append({
            "segment"        : seg,
            "n"              : n,
            "conversions"    : int(conversions),
            "conv_rate"      : round(conv_rate, 4),
            "revenue_inr"    : revenue,
            "roi_pct"        : round(roi, 2),
            "cpa_inr"        : round(cpa, 2),
        })
    seg_df = pd.DataFrame(rows).sort_values("roi_pct", ascending=False)
    print("\n── Segment ROI ───────────────────────────────────────")
    print(seg_df.to_string(index=False))
    return seg_df


def incremental_lift_analysis(df: pd.DataFrame, assumptions: Dict = None) -> Dict:
    """Revenue attributable to campaign vs organic baseline."""
    if assumptions is None:
        assumptions = DEFAULT_ASSUMPTIONS

    actual_rate   = df["y_binary"].mean()
    baseline_rate = assumptions["baseline_conversion_rate"]
    n             = len(df)
    rev_per_conv  = assumptions["revenue_per_conversion"]

    actual_conversions   = df["y_binary"].sum()
    baseline_conversions = baseline_rate * n
    incremental_conv     = max(actual_conversions - baseline_conversions, 0)

    incremental_revenue  = incremental_conv * rev_per_conv
    total_cost           = n * assumptions["cost_per_contact_cellular"]
    incremental_roi      = (incremental_revenue - total_cost) / total_cost * 100

    result = {
        "actual_rate"          : round(actual_rate, 4),
        "baseline_rate"        : baseline_rate,
        "actual_conversions"   : int(actual_conversions),
        "baseline_conversions" : int(baseline_conversions),
        "incremental_conv"     : int(incremental_conv),
        "incremental_revenue"  : incremental_revenue,
        "incremental_roi_pct"  : round(incremental_roi, 2),
    }
    print("\n── Incremental Lift Analysis ─────────────────────────")
    for k, v in result.items():
        print(f"  {k:30s}: {v:,.2f}" if isinstance(v, float) else f"  {k:30s}: {v:,}")
    return result


def sensitivity_analysis(df: pd.DataFrame, assumptions: Dict = None,
                          shocks: list = None) -> pd.DataFrame:
    """What happens to ROI if conversion rate changes by ±X%?"""
    if assumptions is None:
        assumptions = DEFAULT_ASSUMPTIONS
    if shocks is None:
        shocks = [-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30]

    base_rate = df["y_binary"].mean()
    n         = len(df)
    rev       = assumptions["revenue_per_conversion"]
    cost      = n * assumptions["cost_per_contact_cellular"]

    rows = []
    for shock in shocks:
        adj_rate   = base_rate * (1 + shock)
        adj_conv   = adj_rate * n
        adj_rev    = adj_conv * rev
        adj_roi    = (adj_rev - cost) / cost * 100
        rows.append({"shock_pct": shock * 100,
                     "conversion_rate": round(adj_rate, 4),
                     "revenue_inr"    : round(adj_rev),
                     "roi_pct"        : round(adj_roi, 2)})

    sens_df = pd.DataFrame(rows)
    return sens_df


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_channel_roi(channel_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("roi_pct",         "ROI (%)",           ["#2563EB","#F59E0B"]),
        ("cpa_inr",         "Cost Per Acquisition (₹)", ["#2563EB","#F59E0B"]),
        ("conversion_rate", "Conversion Rate",    ["#2563EB","#F59E0B"]),
    ]

    for ax, (col, title, colors) in zip(axes, metrics):
        bars = ax.bar(channel_df.index, channel_df[col], color=colors, width=0.4)
        ax.bar_label(bars,
                     labels=[f"{v:.1f}%" if "rate" in col or "roi" in col else f"₹{v:,.0f}"
                              for v in channel_df[col]],
                     padding=4, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        if "rate" in col or "roi" in col:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.suptitle("Channel ROI Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "channel_roi.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sensitivity(sens_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#EF4444" if v < 0 else "#10B981" for v in sens_df["roi_pct"]]
    bars = ax.bar(sens_df["shock_pct"].astype(str) + "%", sens_df["roi_pct"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.axhline(0, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_xlabel("Conversion Rate Shock (%)")
    ax.set_ylabel("Campaign ROI (%)")
    ax.set_title("Sensitivity Analysis: ROI vs Conversion Rate Change",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_roi_analysis(df: pd.DataFrame, assumptions: Dict = None) -> Dict:
    if assumptions is None:
        assumptions = DEFAULT_ASSUMPTIONS

    channel_df = compute_channel_roi(df, assumptions)
    seg_df     = compute_segment_roi(df, assumptions)
    lift       = incremental_lift_analysis(df, assumptions)
    sens_df    = sensitivity_analysis(df, assumptions)

    plot_channel_roi(channel_df)
    plot_sensitivity(sens_df)

    return {
        "channel_roi"  : channel_df,
        "segment_roi"  : seg_df,
        "lift_analysis": lift,
        "sensitivity"  : sens_df,
    }
