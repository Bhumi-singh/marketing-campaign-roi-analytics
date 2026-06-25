"""
segmentation.py
K-Means customer segmentation for campaign targeting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Select and encode features for clustering."""
    features = ["age", "campaign", "previous", "euribor3m",
                "cons_conf_idx", "cons_price_idx", "emp_var_rate"]

    cat_dummies = pd.get_dummies(df[["job", "education", "marital"]], drop_first=True)
    X = pd.concat([df[features].reset_index(drop=True),
                   cat_dummies.reset_index(drop=True)], axis=1)
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.columns.tolist(), scaler


def elbow_plot(X_scaled, k_range=range(2, 9), figsize=(8, 4)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000, random_state=42))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(k_range, inertias, marker="o", color="#1F4E79", linewidth=2)
    ax1.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")

    ax2.plot(k_range, silhouettes, marker="o", color="#F4A261", linewidth=2)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    plt.tight_layout()
    return fig, inertias, silhouettes


def fit_kmeans(X_scaled, k=4):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    print(f"K={k}  |  Silhouette Score: {score:.4f}")
    return km, labels


def pca_scatter(X_scaled, labels, figsize=(9, 6)):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    colors = ["#1F4E79", "#F4A261", "#2CA02C", "#E76F51"]

    fig, ax = plt.subplots(figsize=figsize)
    for i in np.unique(labels):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i % len(colors)], label=f"Segment {i+1}",
                   alpha=0.4, s=10)
    ax.set_title("Customer Segments — PCA Projection", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend()
    plt.tight_layout()
    return fig


def segment_profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    df = df.copy()
    df["segment"] = labels + 1
    profile = df.groupby("segment").agg(
        size=("subscribed", "count"),
        conv_rate=("subscribed", "mean"),
        avg_age=("age", "mean"),
        avg_contacts=("campaign", "mean"),
        prev_contacted=("was_contacted_before", "mean"),
        euribor=("euribor3m", "mean"),
    ).reset_index()
    profile["conv_rate"] = (profile["conv_rate"] * 100).round(2)
    profile["avg_age"] = profile["avg_age"].round(1)
    profile["avg_contacts"] = profile["avg_contacts"].round(2)
    profile["prev_contacted"] = (profile["prev_contacted"] * 100).round(1)
    profile["share_pct"] = (profile["size"] / profile["size"].sum() * 100).round(1)
    return profile


def segment_conv_bar(profile: pd.DataFrame, figsize=(8, 5)):
    colors = ["#1F4E79", "#F4A261", "#2CA02C", "#E76F51"]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [f"Seg {s}" for s in profile["segment"]],
        profile["conv_rate"],
        color=[colors[i % len(colors)] for i in range(len(profile))],
        width=0.5
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Conversion Rate by Customer Segment", fontsize=13, fontweight="bold")
    ax.set_ylabel("Conversion Rate (%)")
    for bar, (_, row) in zip(bars, profile.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{row['conv_rate']:.1f}%\n(n={row['size']:,})",
                ha="center", fontsize=9)
    plt.tight_layout()
    return fig
