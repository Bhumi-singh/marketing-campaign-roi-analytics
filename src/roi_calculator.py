"""
roi_calculator.py
Campaign ROI measurement, CPA calculation, and sensitivity analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def compute_roi(n_contacts: int, n_converted: int,
                cost_per_contact: float = 50,
                revenue_per_conversion: float = 5000) -> dict:
    total_cost = n_contacts * cost_per_contact
    total_revenue = n_converted * revenue_per_conversion
    profit = total_revenue - total_cost
    roi = profit / total_cost * 100
    cpa = total_cost / n_converted if n_converted > 0 else np.inf
    conv_rate = n_converted / n_contacts * 100

    return {
        "n_contacts": n_contacts,
        "n_converted": n_converted,
        "conv_rate_pct": round(conv_rate, 2),
        "total_cost": round(total_cost, 2),
        "total_revenue": round(total_revenue, 2),
        "profit": round(profit, 2),
        "roi_pct": round(roi, 2),
        "cpa": round(cpa, 2),
        "cost_per_contact": cost_per_contact,
        "revenue_per_conversion": revenue_per_conversion,
    }


def print_roi_report(roi: dict, label="Campaign ROI Report"):
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Contacts Made         : {roi['n_contacts']:,}")
    print(f"  Conversions           : {roi['n_converted']:,}")
    print(f"  Conversion Rate       : {roi['conv_rate_pct']:.2f}%")
    print(f"  Cost Per Contact      : ₹{roi['cost_per_contact']:,.0f}")
    print(f"  Total Cost            : ₹{roi['total_cost']:,.0f}")
    print(f"  Revenue Per Conversion: ₹{roi['revenue_per_conversion']:,.0f}")
    print(f"  Total Revenue         : ₹{roi['total_revenue']:,.0f}")
    print(f"  Profit                : ₹{roi['profit']:,.0f}")
    print(f"  ROI                   : {roi['roi_pct']:.1f}%")
    print(f"  Cost Per Acquisition  : ₹{roi['cpa']:,.0f}")
    print(f"{'='*52}\n")


def channel_roi_comparison(df, cost_per_contact=50, revenue_per_conversion=5000):
    results = []
    for channel, grp in df.groupby("contact"):
        r = compute_roi(len(grp), grp["subscribed"].sum(),
                        cost_per_contact, revenue_per_conversion)
        r["channel"] = channel
        results.append(r)
    return pd.DataFrame(results).sort_values("roi_pct", ascending=False)


def segment_roi(df, cost_per_contact=50, revenue_per_conversion=5000):
    results = []
    seg_col = "segment_name" if "segment_name" in df.columns else "segment"
    for seg, grp in df.groupby(seg_col):
        r = compute_roi(len(grp), grp["subscribed"].sum(),
                        cost_per_contact, revenue_per_conversion)
        r["segment"] = seg
        results.append(r)
    return pd.DataFrame(results).sort_values("roi_pct", ascending=False)


def incremental_lift_roi(control_n, control_conv, treatment_n, treatment_conv,
                         cost_per_contact=50, revenue_per_conversion=5000):
    baseline_rate = control_conv / control_n
    treatment_rate = treatment_conv / treatment_n
    incremental_convs = (treatment_rate - baseline_rate) * treatment_n
    incremental_revenue = incremental_convs * revenue_per_conversion
    cost = treatment_n * cost_per_contact
    incremental_roi = incremental_revenue / cost * 100
    return {
        "baseline_rate": round(baseline_rate * 100, 2),
        "treatment_rate": round(treatment_rate * 100, 2),
        "lift_pp": round((treatment_rate - baseline_rate) * 100, 2),
        "incremental_conversions": round(incremental_convs),
        "incremental_revenue": round(incremental_revenue, 2),
        "total_cost": round(cost, 2),
        "incremental_roi": round(incremental_roi, 2),
    }


def sensitivity_analysis(base_conv_rate, n_contacts=10000,
                          cost_per_contact=50, revenue_per_conversion=5000,
                          conv_range=(-50, 50), steps=21):
    """How does ROI change as conversion rate varies ±N%?"""
    pct_changes = np.linspace(conv_range[0], conv_range[1], steps)
    rows = []
    for pct in pct_changes:
        adj_rate = base_conv_rate * (1 + pct / 100)
        r = compute_roi(n_contacts, int(n_contacts * adj_rate),
                        cost_per_contact, revenue_per_conversion)
        r["conv_change_pct"] = pct
        rows.append(r)
    return pd.DataFrame(rows)


def plot_channel_roi(channel_df, figsize=(9, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = ["#1F4E79", "#9DC3E6"]

    for ax, col, label in zip(axes,
        ["roi_pct", "cpa"],
        ["ROI (%)", "Cost Per Acquisition (₹)"]):
        bars = ax.bar(channel_df["channel"], channel_df[col],
                      color=colors[:len(channel_df)], width=0.4, alpha=0.9)
        ax.set_title(f"Channel {label}", fontweight="bold")
        ax.set_ylabel(label)
        for bar, val in zip(bars, channel_df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(channel_df[col])*0.02,
                    f"{val:,.0f}", ha="center", fontsize=10, fontweight="bold")
    plt.suptitle("Channel-Level ROI Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_sensitivity(sens_df, figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sens_df["conv_change_pct"], sens_df["roi_pct"],
            color="#1F4E79", linewidth=2.5, marker="o", markersize=4)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Break-even")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, label="Base scenario")
    ax.fill_between(sens_df["conv_change_pct"], sens_df["roi_pct"],
                    where=sens_df["roi_pct"] > 0, alpha=0.15, color="#1F4E79", label="Profitable")
    ax.fill_between(sens_df["conv_change_pct"], sens_df["roi_pct"],
                    where=sens_df["roi_pct"] <= 0, alpha=0.15, color="red", label="Loss")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("ROI Sensitivity to Conversion Rate Change", fontsize=13, fontweight="bold")
    ax.set_xlabel("Change in Conversion Rate (%)")
    ax.set_ylabel("Campaign ROI (%)")
    ax.legend()
    plt.tight_layout()
    return fig
