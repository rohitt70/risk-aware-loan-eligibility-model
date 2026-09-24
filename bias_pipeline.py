"""
bias_pipeline.py
-----------------
Core ML pipeline for Bias-Aware Loan Eligibility Prediction.
Handles both datasets:
  - German Credit Risk (german_credit_risk.csv)
  - Loan Dataset (loan_train_cleaned.csv + loan_test_cleaned.csv)

Models: Logistic Regression, Random Forest, XGBoost
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from fairness_metrics import (
    compute_fairness_metrics, fairness_summary,
    reweigh_samples, threshold_optimization
)

# ──────────────────────────────────────────────────────────────────────────────
# GERMAN CREDIT RISK DATASET
# ──────────────────────────────────────────────────────────────────────────────

GERMAN_PROTECTED = ["Sex", "Age"]   # protected attributes
GERMAN_TARGET    = "Risk"            # good / bad

def load_german(path="german_credit_risk.csv"):
    df = pd.read_csv(path)
    df["Risk_bin"] = (df[GERMAN_TARGET] == "good").astype(int)
    return df


def preprocess_german(df):
    """Encode categoricals, return X, y, encoders."""
    feature_cols = [c for c in df.columns if c not in [GERMAN_TARGET, "Risk_bin"]]
    df_enc = df[feature_cols].copy()

    encoders = {}
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X = scaler.fit_transform(df_enc)
    y = df["Risk_bin"].values
    return X, y, feature_cols, encoders, scaler, df_enc


# ──────────────────────────────────────────────────────────────────────────────
# LOAN DATASET
# ──────────────────────────────────────────────────────────────────────────────

LOAN_PROTECTED = ["Gender", "Married", "Education"]
LOAN_TARGET    = "Loan_Status"
LOAN_DROP      = ["Loan_ID"]

def load_loan_train(path="loan_train_cleaned.csv"):
    df = pd.read_csv(path)
    df["Loan_bin"] = (df[LOAN_TARGET] == "Y").astype(int)
    return df


def load_loan_test(path="loan_test_cleaned.csv"):
    return pd.read_csv(path)


def preprocess_loan(train_df, test_df=None):
    """Encode loan training data; optionally transform test set."""
    drop_cols = LOAN_DROP + [LOAN_TARGET, "Loan_bin"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["Loan_bin"].values

    encoders = {}
    for col in X_train.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    X_test_sc = None
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        for col, le in encoders.items():
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(str).map(
                    lambda v, le=le: le.transform([v])[0]
                    if v in le.classes_ else -1
                )
        X_test_sc = scaler.transform(X_test)

    return X_train_sc, y_train, feature_cols, encoders, scaler, X_test_sc, X_train


# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        )
    return models


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       sensitive_col, df_train, df_test,
                       sample_weight=None):
    """
    Train all models, evaluate performance + fairness metrics.
    Returns dict: {model_name: {metrics, fairness, model}}
    """
    results = {}
    models = build_models()

    for name, model in models.items():
        print(f"  Training {name}...")
        if sample_weight is not None and hasattr(model, "fit"):
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # ── Performance ────────────────────────────────────────────────────
        perf = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
            "Classification Report": classification_report(y_test, y_pred, output_dict=True),
        }

        # ── Fairness ────────────────────────────────────────────────────────
        fairness = compute_fairness_metrics(y_test, y_pred, sensitive_col, df_test)

        # ── Per-group threshold optimization ───────────────────────────────
        opt_thresholds = None
        if y_prob is not None:
            opt_thresholds = threshold_optimization(y_prob, y_test, sensitive_col, df_test)

        results[name] = {
            "model":              model,
            "performance":        perf,
            "fairness":           fairness,
            "opt_thresholds":     opt_thresholds,
            "y_pred":             y_pred,
            "y_prob":             y_prob,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE RUNNERS
# ──────────────────────────────────────────────────────────────────────────────

def run_german_pipeline():
    """End-to-end pipeline for German Credit dataset."""
    from sklearn.model_selection import train_test_split

    print("\n[INFO] Loading German Credit Risk dataset...")
    df = load_german()
    X, y, feature_cols, encoders, scaler, df_enc = preprocess_german(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    df_train = df.loc[idx_train].reset_index(drop=True)
    df_test  = df.loc[idx_test].reset_index(drop=True)

    # ── Compute reweighing weights ──────────────────────────────────────────
    print("  Computing reweighing sample weights (Sex)...")
    weights = reweigh_samples(df_train, sensitive_col="Sex", label_col="Risk_bin")

    print("  Training without bias mitigation...")
    results_base = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        sensitive_col="Sex", df_train=df_train, df_test=df_test,
        sample_weight=None
    )

    print("  Training WITH reweighing (bias mitigation)...")
    results_fair = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        sensitive_col="Sex", df_train=df_train, df_test=df_test,
        sample_weight=weights
    )

    return {
        "dataset":      "German Credit Risk",
        "df":           df,
        "df_test":      df_test,
        "feature_cols": feature_cols,
        "protected":    GERMAN_PROTECTED,
        "base":         results_base,
        "fair":         results_fair,
    }


def run_loan_pipeline():
    """End-to-end pipeline for Loan dataset."""
    print("\n[INFO] Loading Loan dataset...")
    train_df = load_loan_train()
    test_df  = load_loan_test()

    X_train, y_train, feature_cols, encoders, scaler, X_test_raw, X_train_df = \
        preprocess_loan(train_df, test_df)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
        X_train, y_train, train_df.index, test_size=0.2, random_state=42, stratify=y_train
    )

    df_tr  = train_df.loc[idx_tr].reset_index(drop=True)
    df_val = train_df.loc[idx_val].reset_index(drop=True)

    print("  Computing reweighing sample weights (Gender)...")
    weights = reweigh_samples(df_tr, sensitive_col="Gender", label_col="Loan_bin")

    print("  Training without bias mitigation...")
    results_base = train_and_evaluate(
        X_tr, y_tr, X_val, y_val,
        sensitive_col="Gender", df_train=df_tr, df_test=df_val,
        sample_weight=None
    )

    print("  Training WITH reweighing (bias mitigation)...")
    results_fair = train_and_evaluate(
        X_tr, y_tr, X_val, y_val,
        sensitive_col="Gender", df_train=df_tr, df_test=df_val,
        sample_weight=weights
    )

    return {
        "dataset":      "Loan Eligibility",
        "df":           train_df,
        "df_test":      df_val,
        "feature_cols": feature_cols,
        "protected":    LOAN_PROTECTED,
        "base":         results_base,
        "fair":         results_fair,
    }


if __name__ == "__main__":
    r1 = run_german_pipeline()
    print("\n=== German Credit – Base Model (Logistic Regression) ===")
    print(r1["base"]["Logistic Regression"]["performance"])
    print(r1["base"]["Logistic Regression"]["fairness"])

    r2 = run_loan_pipeline()
    print("\n=== Loan Dataset – Base Model (Logistic Regression) ===")
    print(r2["base"]["Logistic Regression"]["performance"])
    print(r2["base"]["Logistic Regression"]["fairness"])
