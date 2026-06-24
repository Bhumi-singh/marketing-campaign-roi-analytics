"""
model.py
Logistic Regression + XGBoost conversion predictor with SHAP explainability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

FIG_DIR   = Path("reports/figures")
MODEL_DIR = Path("models")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Features that go into the model (exclude raw text columns & target)
EXCLUDE = {
    "y", "y_binary", "age_group", "month", "day_of_week",
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "poutcome", "segment_label",
}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE and df[c].dtype in [np.float64, np.int64, int, float]]


def prepare_data(df: pd.DataFrame, test_size: float = 0.2
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df["y_binary"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


# ── Logistic Regression ───────────────────────────────────────────────────────

def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced",
                                       solver="lbfgs", random_state=42))
    ])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"[LogReg] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    pipe.fit(X_train, y_train)
    return pipe


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train) -> XGBClassifier:
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"[XGBoost] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    xgb.fit(X_train, y_train)
    return xgb


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str) -> Dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.35).astype(int)          # lower threshold due to class imbalance

    auc  = roc_auc_score(y_test, y_prob)
    rep  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n── {name} Test Performance ─────────────────────")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No","Yes"]))

    return {"model": name, "auc": auc, "report": rep, "y_prob": y_prob}


def plot_roc_pr(lr_result, xgb_result, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for res, color in [(lr_result,"#2563EB"), (xgb_result,"#10B981")]:
        fpr_tpr = _roc_points(y_test, res["y_prob"])
        ax1.plot(fpr_tpr[0], fpr_tpr[1], color=color,
                 label=f"{res['model']} (AUC={res['auc']:.3f})", linewidth=2)
    ax1.plot([0,1],[0,1],"--",color="#6B7280")
    ax1.set_title("ROC Curve"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.legend()

    for res, color in [(lr_result,"#2563EB"), (xgb_result,"#10B981")]:
        prec, rec = _pr_points(y_test, res["y_prob"])
        ax2.plot(rec, prec, color=color, label=res["model"], linewidth=2)
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend()

    plt.suptitle("Model Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def _roc_points(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def _pr_points(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return prec, rec


# ── SHAP ──────────────────────────────────────────────────────────────────────

def explain_xgboost(xgb_model: XGBClassifier, X_test: pd.DataFrame, n_samples: int = 500):
    X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_sample)

    # Summary beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.title("SHAP Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Bar chart top-10
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False, max_display=10)
    plt.title("Top 10 SHAP Features", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(FIG_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()

    return explainer, shap_vals


# ── Scored list ───────────────────────────────────────────────────────────────

def score_customers(df: pd.DataFrame, xgb_model: XGBClassifier) -> pd.DataFrame:
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(df[feat_cols].median())
    df = df.copy()
    df["conversion_probability"] = xgb_model.predict_proba(X)[:, 1]
    df["priority_rank"]          = df["conversion_probability"].rank(ascending=False).astype(int)
    return df.sort_values("conversion_probability", ascending=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_modeling(df: pd.DataFrame) -> Dict:
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class balance (train): {y_train.mean():.2%} positive")

    lr_model  = train_logistic(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    lr_result  = evaluate_model(lr_model,  X_test, y_test, "Logistic Regression")
    xgb_result = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    plot_roc_pr(lr_result, xgb_result, y_test)
    explain_xgboost(xgb_model, X_test)

    # Save best model
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(lr_model,  MODEL_DIR / "logreg_model.pkl")
    print(f"\n✅ Models saved to {MODEL_DIR}/")

    return {
        "lr_model"     : lr_model,
        "xgb_model"    : xgb_model,
        "lr_result"    : lr_result,
        "xgb_result"   : xgb_result,
        "feature_cols" : get_feature_cols(df),
        "X_test"       : X_test,
        "y_test"       : y_test,
    }
