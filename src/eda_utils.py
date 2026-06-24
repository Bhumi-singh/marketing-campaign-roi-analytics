"""
eda_utils.py
Reusable plotting and summary helpers for EDA notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

PALETTE = ["#1F4E79", "#2E75B6", "#9DC3E6", "#F4A261", "#E76F51"]
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")


def conversion_rate_bar(df, col, title=None, top_n=None, figsize=(10, 5)):
    """Bar chart of conversion rate (%) by a categorical column."""
    grp = df.groupby(col)["subscribed"].agg(["mean", "count"]).reset_index()
    grp.columns = [col, "conv_rate", "n"]
    grp["conv_rate"] *= 100
    grp = grp.sort_values("conv_rate", ascending=False)
    if top_n:
        grp = grp.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(grp[col].astype(str), grp["conv_rate"], color=PALETTE[0], alpha=0.85)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(title or f"Conversion Rate by {col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Conversion Rate (%)")
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{row['conv_rate']:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def distribution_plot(df, col, hue="subscribed", figsize=(10, 5)):
    """KDE distribution split by target."""
    fig, ax = plt.subplots(figsize=figsize)
    for val, label, color in zip([0, 1], ["Not Subscribed", "Subscribed"], [PALETTE[3], PALETTE[0]]):
        subset = df[df[hue] == val][col].dropna()
        subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)
    ax.set_title(f"Distribution of {col} by Campaign Outcome", fontsize=13, fontweight="bold")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.legend()
    plt.tight_layout()
    return fig


def funnel_chart(stages: dict, figsize=(8, 5)):
    """Horizontal funnel chart."""
    labels = list(stages.keys())
    values = list(stages.values())
    pcts = [v / values[0] * 100 for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("Blues_d", len(labels))
    bars = ax.barh(labels[::-1], values[::-1], color=colors)
    for bar, val, pct in zip(bars, values[::-1], pcts[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va="center", fontsize=10)
    ax.set_title("Campaign Funnel", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(values) * 1.25)
    plt.tight_layout()
    return fig


def correlation_heatmap(df, cols, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
                ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Quick summary: conversion rate, count, share of total per group."""
    total = len(df)
    converted = df["subscribed"].sum()
    print(f"\n{'='*45}")
    print(f"  Total Records   : {total:,}")
    print(f"  Total Converted : {converted:,}  ({converted/total*100:.2f}%)")
    print(f"{'='*45}\n")
    return df.describe(include="all")
