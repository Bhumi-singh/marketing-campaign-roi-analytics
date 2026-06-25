"""
model.py
Conversion probability prediction using Logistic Regression and XGBoost.
Includes SHAP-based feature importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, average_precision_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def encode_features(df: pd.DataFrame):
    """Encode categoricals and return X, y."""
    drop_cols = ["subscribed", "segment", "segment_name", "age_group", "season"]
    target = "subscribed"

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

    # Encode all remaining object columns
    for col in X.select_dtypes("object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X.astype(float), y


def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    return lr, xgb_model


def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Sub","Subscribed"]))
    return auc, ap, y_prob


def plot_roc_curves(y_test, probs_dict, figsize=(8, 6)):
    colors = ["#1F4E79", "#F4A261"]
    fig, ax = plt.subplots(figsize=figsize)
    for (name, y_prob), color in zip(probs_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random")
    ax.set_title("ROC Curves — Conversion Prediction Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_precision_recall(y_test, probs_dict, figsize=(8, 6)):
    colors = ["#1F4E79", "#F4A261"]
    fig, ax = plt.subplots(figsize=figsize)
    for (name, y_prob), color in zip(probs_dict.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{name} (AP={ap:.3f})")
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.tight_layout()
    return fig


def shap_analysis(xgb_model, X_test, feature_names, top_n=15, figsize=(9, 7)):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        mean_abs[idx[::-1]],
        color="#1F4E79", alpha=0.85
    )
    ax.set_title(f"SHAP Feature Importance — Top {top_n} Drivers", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    return fig, shap_values


def score_customers(xgb_model, X, df_original) -> pd.DataFrame:
    """Score all customers and return ranked list."""
    probs = xgb_model.predict_proba(X)[:, 1]
    scored = df_original.copy()
    scored["conv_probability"] = probs
    scored["priority_rank"] = scored["conv_probability"].rank(ascending=False).astype(int)
    return scored.sort_values("conv_probability", ascending=False)
