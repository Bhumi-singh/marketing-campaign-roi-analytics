"""
segmentation.py
K-Means customer segmentation on the bank marketing dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Features used for clustering
SEG_FEATURES = [
    "age",
    "duration",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "was_previously_contacted",
    "month_num",
]

SEGMENT_LABELS = {
    0: "High-Value Responders",
    1: "At-Risk / Low-Engagement",
    2: "Price-Sensitive Fence-Sitters",
    3: "Loyal Previously-Contacted",
}


def build_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean feature matrix for clustering."""
    feat_cols = [c for c in SEG_FEATURES if c in df.columns]
    X = df[feat_cols].copy()
    X = X.fillna(X.median())
    return X


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> int:
    """Elbow + silhouette to pick k."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "o-", color="#2563EB")
    ax1.set_title("Elbow Method"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")

    ax2.plot(list(k_range), silhouettes, "o-", color="#10B981")
    ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.set_ylabel("Score")

    plt.suptitle("Optimal k Selection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "optimal_k.png", dpi=150, bbox_inches="tight")
    plt.show()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"Best k by silhouette: {best_k}  (score={max(silhouettes):.3f})")
    return best_k


def fit_kmeans(df: pd.DataFrame, k: int = 4) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    X    = build_segment_features(df)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    df = df.copy()
    df["segment"]       = km.fit_predict(X_sc)
    df["segment_label"] = df["segment"].map(SEGMENT_LABELS)

    sil = silhouette_score(X_sc, df["segment"])
    print(f"[KMeans k={k}] Silhouette = {sil:.3f}")
    return df, km, scaler


def segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profile_cols = [
        "age", "duration", "campaign", "previous",
        "emp.var.rate", "euribor3m", "y_binary",
    ]
    profile = (
        df.groupby("segment_label")[profile_cols]
        .agg(["mean", "std"])
        .round(2)
    )
    # Flat columns
    profile.columns = ["_".join(c) for c in profile.columns]
    profile["n"] = df.groupby("segment_label").size()
    profile["conversion_rate"] = df.groupby("segment_label")["y_binary"].mean().round(4)
    profile = profile.sort_values("conversion_rate", ascending=False)
    print("\nSegment Profiles:")
    print(profile[["n","conversion_rate","age_mean","duration_mean","campaign_mean"]].to_string())
    return profile


def plot_segment_radar(df: pd.DataFrame):
    """Radar chart comparing segments across key features."""
    from matplotlib.patches import FancyArrowPatch
    features = ["age","duration","campaign","previous","euribor3m"]
    seg_means = df.groupby("segment_label")[features].mean()

    # Normalise 0-1 per feature
    seg_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#2563EB","#EF4444","#F59E0B","#10B981"]

    for (label, row), color in zip(seg_norm.iterrows(), colors):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_title("Segment Radar Chart", size=14, fontweight="bold", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "segment_radar.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_segment_conversion(df: pd.DataFrame):
    rates = df.groupby("segment_label")["y_binary"].mean().sort_values(ascending=True) * 100
    colors = ["#EF4444","#F59E0B","#6366F1","#10B981"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(rates.index, rates.values, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontweight="bold")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_title("Conversion Rate by Customer Segment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "segment_conversion.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_segmentation(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X      = build_segment_features(df)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    find_optimal_k(X_sc)
    df_seg, km, _ = fit_kmeans(df, k=k)
    profiles       = segment_profiles(df_seg)
    plot_segment_radar(df_seg)
    plot_segment_conversion(df_seg)

    out = "data/processed/segments.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_csv(out, index=False)
    print(f"\n✅ Segmented data saved → {out}")
    return df_seg
