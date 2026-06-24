"""
model.py — Conversion prediction: Logistic Regression + XGBoost + SHAP.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier
import shap
 
 
# ─── Preprocessing ──────────────────────────────────────────────────────────
 
CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]
 
def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Encode categoricals, drop non-predictive cols, return X, y, feature_names.
    """
    df = df.copy()
 
    # Drop leaky / non-useful columns
    drop_cols = ["duration"]          # duration is post-hoc leakage
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
 
    # Encode target
    if df["y"].dtype == object:
        df["y"] = (df["y"] == "yes").astype(int)
 
    # Label-encode categoricals
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
 
    # day_of_week if present
    if "day_of_week" in df.columns:
        df["day_of_week"] = le.fit_transform(df["day_of_week"].astype(str))
 
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y, X.columns.tolist()
 
 
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)
 
 
# ─── Logistic Regression ────────────────────────────────────────────────────
 
def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_train)
    return lr, scaler
 
 
def evaluate_logistic(lr, scaler, X_test, y_test):
    X_sc = scaler.transform(X_test)
    y_pred = lr.predict(X_sc)
    y_prob = lr.predict_proba(X_sc)[:, 1]
 
    print("=== Logistic Regression ===")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision: {average_precision_score(y_test, y_prob):.4f}")
    return y_prob
 
 
# ─── XGBoost ────────────────────────────────────────────────────────────────
 
def train_xgboost(X_train, y_train):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False)
    return xgb
 
 
def evaluate_xgboost(xgb, X_test, y_test):
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
 
    print("=== XGBoost ===")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision: {average_precision_score(y_test, y_prob):.4f}")
    return y_prob
 
 
# ─── Visualisation ───────────────────────────────────────────────────────────
 
def plot_roc_curves(y_test, lr_prob, xgb_prob, save_path=None):
    plt.figure(figsize=(7, 5))
    for prob, label in [(lr_prob, "Logistic Regression"), (xgb_prob, "XGBoost")]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_precision_recall(y_test, lr_prob, xgb_prob, save_path=None):
    plt.figure(figsize=(7, 5))
    for prob, label in [(lr_prob, "Logistic Regression"), (xgb_prob, "XGBoost")]:
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_xgb_feature_importance(xgb, feature_names, top_n=15, save_path=None):
    imp = pd.Series(xgb.feature_importances_, index=feature_names)
    imp = imp.nlargest(top_n).sort_values()
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="steelblue")
    plt.title(f"XGBoost Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ─── SHAP Analysis ──────────────────────────────────────────────────────────
 
def compute_shap_values(xgb, X_test):
    """Return SHAP explainer and values for the XGBoost model."""
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values
 
 
def plot_shap_summary(shap_values, X_test, save_path=None):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary — Feature Impact on Conversion")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_shap_bar(shap_values, X_test, save_path=None):
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Mean |SHAP| — Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def plot_shap_waterfall(explainer, X_test, idx=0, save_path=None):
    """Waterfall plot for a single prediction."""
    shap.plots.waterfall(explainer(X_test)[idx], show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ─── Scored Customer List ────────────────────────────────────────────────────
 
def score_customers(df_original: pd.DataFrame, xgb, X, feature_names: list) -> pd.DataFrame:
    """
    Return original dataframe enriched with conversion probability scores,
    ranked highest-first.
    """
    probs = xgb.predict_proba(X[feature_names])[:, 1]
    df_scored = df_original.copy()
    df_scored["conversion_probability"] = probs
    df_scored["score_rank"] = df_scored["conversion_probability"].rank(
        ascending=False, method="first"
    ).astype(int)
    return df_scored.sort_values("conversion_probability", ascending=False)
 
 
def save_scored_list(df_scored: pd.DataFrame, path="data/processed/scored_customers.csv"):
    df_scored.to_csv(path, index=False)
    print(f"Scored customer list saved → {path}  ({len(df_scored):,} rows)")
 
 
# ─── Cross-Validation ────────────────────────────────────────────────────────
 
def cross_validate_model(model, X, y, cv=5, scoring="roc_auc"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    print(f"Cross-Val {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores
 