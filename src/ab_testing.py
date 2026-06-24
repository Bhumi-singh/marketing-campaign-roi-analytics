"""
ab_testing.py
Statistical significance testing for campaign channel and treatment comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def proportions_ztest(n_A, conv_A, n_B, conv_B, alpha=0.05):
    """
    Two-proportion z-test.
    Returns: z_stat, p_value, is_significant, lift_pct, confidence_interval
    """
    p_A = conv_A / n_A
    p_B = conv_B / n_B
    p_pool = (conv_A + conv_B) / (n_A + n_B)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
    z = (p_B - p_A) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    lift = (p_B - p_A) / p_A * 100

    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt(p_A*(1-p_A)/n_A + p_B*(1-p_B)/n_B)
    ci_low = (p_B - p_A) - z_crit * se_diff
    ci_high = (p_B - p_A) + z_crit * se_diff

    return {
        "p_control": round(p_A * 100, 2),
        "p_treatment": round(p_B * 100, 2),
        "z_statistic": round(z, 4),
        "p_value": round(p_val, 6),
        "is_significant": p_val < alpha,
        "lift_pct": round(lift, 2),
        "ci_95": (round(ci_low * 100, 3), round(ci_high * 100, 3)),
        "alpha": alpha,
    }


def chi_square_test(df, group_col, target_col="subscribed"):
    """Chi-square test of independence between a categorical column and the target."""
    ct = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "dof": dof,
        "is_significant": p < 0.05,
        "contingency_table": ct,
    }


def ttest_by_group(df, group_col, value_col, group_a, group_b):
    """Independent t-test comparing a numeric column between two groups."""
    a = df[df[group_col] == group_a][value_col].dropna()
    b = df[df[group_col] == group_b][value_col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "group_a": group_a, "mean_a": round(a.mean(), 4),
        "group_b": group_b, "mean_b": round(b.mean(), 4),
        "t_statistic": round(t, 4),
        "p_value": round(p, 6),
        "is_significant": p < 0.05,
    }


def print_ab_report(result: dict, label="A/B Test Result"):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Control conversion rate   : {result['p_control']}%")
    print(f"  Treatment conversion rate : {result['p_treatment']}%")
    print(f"  Lift                      : {result['lift_pct']:+.2f}%")
    print(f"  Z-statistic               : {result['z_statistic']}")
    print(f"  P-value                   : {result['p_value']}")
    print(f"  95% CI (diff)             : {result['ci_95']}")
    sig = "✅ SIGNIFICANT" if result["is_significant"] else "❌ NOT significant"
    print(f"  Result                    : {sig}  (α={result['alpha']})")
    print(f"{'='*50}\n")


def plot_conversion_comparison(groups: dict, title="Conversion Rate by Group", figsize=(8, 5)):
    """
    groups = {"Cellular": 0.147, "Telephone": 0.052}
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1F4E79", "#9DC3E6", "#F4A261", "#E76F51"]
    labels = list(groups.keys())
    values = [v * 100 for v in groups.values()]
    bars = ax.bar(labels, values, color=colors[:len(labels)], width=0.5, alpha=0.9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Conversion Rate (%)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_lift_waterfall(baseline_rate, treatment_rate, figsize=(7, 4)):
    """Waterfall showing baseline → lift → treatment rate."""
    lift = treatment_rate - baseline_rate
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["Baseline\n(Telephone)"], [baseline_rate * 100], color="#9DC3E6", width=0.4)
    ax.bar(["Lift\n(Incremental)"], [lift * 100], bottom=[baseline_rate * 100],
           color="#1F4E79", width=0.4)
    ax.bar(["Treatment\n(Cellular)"], [treatment_rate * 100], color="#1F4E79", alpha=0.7, width=0.4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Incremental Lift: Cellular vs Telephone", fontsize=13, fontweight="bold")
    ax.set_ylabel("Conversion Rate (%)")
    plt.tight_layout()
    return fig
