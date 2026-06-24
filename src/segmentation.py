"""
segmentation.py — Customer segmentation using K-Means clustering.
"""
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
 
# ─── Feature Engineering ────────────────────────────────────────────────────
 
def engineer_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features suitable for customer segmentation."""
    df = df.copy()
 
    # Age bands
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    )
 
    # Balance bands
    df["balance_band"] = pd.cut(
        df["balance"],
        bins=[-10000, 0, 500, 2000, 10000, 200000],
        labels=["negative", "low", "medium", "high", "very_high"]
    )
 
    # Contact recency proxy (pdays: -1 means never contacted)
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    df["recency_days"] = df["pdays"].replace(-1, 999)  # 999 = never
 
    # Frequency of contact in this campaign
    df["high_contact"] = (df["campaign"] > df["campaign"].median()).astype(int)
 
    return df
 
 
def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and encode numeric features for clustering."""
    features = [
        "age", "balance", "duration", "campaign",
        "pdays", "previous", "previously_contacted", "high_contact"
    ]
    return df[features].fillna(0)
 
 
# ─── Optimal K Selection ────────────────────────────────────────────────────
 
def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> dict:
    """Compute inertia and silhouette scores for each k."""
    results = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(X_scaled, labels))
    return results
 
 
def plot_elbow_silhouette(results: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].plot(results["k"], results["inertia"], "bo-")
    axes[0].set_title("Elbow Method — Inertia vs K")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
 
    axes[1].plot(results["k"], results["silhouette"], "rs-")
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ─── Fit K-Means ────────────────────────────────────────────────────────────
 
def fit_kmeans(df: pd.DataFrame, k: int = 4) -> tuple:
    """
    Fit K-Means on engineered features.
    Returns: (df_with_segment, scaler, kmeans_model, feature_matrix)
    """
    df = engineer_segment_features(df)
    X = select_clustering_features(df)
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["segment"] = km.fit_predict(X_scaled)
 
    return df, scaler, km, X_scaled
 
 
# ─── Segment Profiling ───────────────────────────────────────────────────────
 
SEGMENT_LABELS = {
    # Will be overwritten after inspecting cluster profiles
    0: "Segment 0",
    1: "Segment 1",
    2: "Segment 2",
    3: "Segment 3",
}
 
def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary profile for each segment.
    Requires 'segment' and 'y' columns (y = converted).
    """
    profile = df.groupby("segment").agg(
        count=("segment", "count"),
        avg_age=("age", "mean"),
        avg_balance=("balance", "mean"),
        avg_duration=("duration", "mean"),
        avg_campaign=("campaign", "mean"),
        conversion_rate=("y", "mean"),
    ).reset_index()
 
    profile["conversion_rate_pct"] = (profile["conversion_rate"] * 100).round(2)
    profile["avg_age"] = profile["avg_age"].round(1)
    profile["avg_balance"] = profile["avg_balance"].round(0)
    profile["avg_duration"] = profile["avg_duration"].round(0)
    return profile
 
 
def label_segments(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-label segments based on conversion rate and balance rank.
    High conversion + high balance → 'High-Value Responders'
    Low conversion + low balance  → 'Price-Sensitive / Hard to Convert'
    High balance, low conversion  → 'Dormant High-Balance'
    Low balance, high contact     → 'Over-Contacted / Fatigued'
    """
    profile = profile.copy().sort_values("conversion_rate_pct", ascending=False)
    labels_map = {}
    preset = [
        "High-Value Responders",
        "Moderate Potential",
        "Dormant High-Balance",
        "Hard-to-Convert",
    ]
    for i, (_, row) in enumerate(profile.iterrows()):
        labels_map[int(row["segment"])] = preset[i] if i < len(preset) else f"Segment {i}"
 
    df["segment_label"] = df["segment"].map(labels_map)
    return df, labels_map
 
 
# ─── Visualisation ───────────────────────────────────────────────────────────
 
def plot_segment_profiles(profile: pd.DataFrame, labels_map: dict, save_path: str = None):
    profile = profile.copy()
    profile["label"] = profile["segment"].map(labels_map)
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = sns.color_palette("Set2", len(profile))
 
    # Conversion rate
    axes[0].bar(profile["label"], profile["conversion_rate_pct"], color=colors)
    axes[0].set_title("Conversion Rate by Segment (%)")
    axes[0].set_ylabel("Conversion Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)
 
    # Average Balance
    axes[1].bar(profile["label"], profile["avg_balance"], color=colors)
    axes[1].set_title("Average Balance by Segment (€)")
    axes[1].set_ylabel("Average Balance")
    axes[1].tick_params(axis="x", rotation=30)
 
    # Segment size
    axes[2].pie(
        profile["count"],
        labels=profile["label"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )
    axes[2].set_title("Segment Size Distribution")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_segment_heatmap(profile: pd.DataFrame, labels_map: dict, save_path: str = None):
    profile = profile.copy()
    profile["label"] = profile["segment"].map(labels_map)
    heat_cols = ["avg_age", "avg_balance", "avg_duration", "avg_campaign", "conversion_rate_pct"]
    heat_data = profile.set_index("label")[heat_cols]
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
 
    plt.figure(figsize=(10, 4))
    sns.heatmap(heat_norm, annot=heat_data.round(1), fmt="g", cmap="YlOrRd", linewidths=0.5)
    plt.title("Segment Feature Heatmap (Normalised)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_conversion_by_segment_job(df: pd.DataFrame, save_path: str = None):
    ct = df.groupby(["segment_label", "job"])["y"].mean().reset_index()
    ct.columns = ["Segment", "Job", "Conversion Rate"]
    pivot = ct.pivot(index="Job", columns="Segment", values="Conversion Rate")
 
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", linewidths=0.3)
    plt.title("Conversion Rate by Segment × Job")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ─── Save Segments ───────────────────────────────────────────────────────────
 
def save_segments(df: pd.DataFrame, path: str = "data/processed/segments.csv"):
    df.to_csv(path, index=False)
    print(f"Segments saved → {path}  ({len(df):,} rows)")