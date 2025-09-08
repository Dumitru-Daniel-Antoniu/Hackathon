# train_booking_risk_baseline.py
# ------------------------------------------------------------
# 1) Loads Training_Data.csv
# 2) Preprocesses features (numeric + categorical)
# 3) Trains Logistic Regression (balanced)
# 4) Evaluates on a validation split (10% of training)
# 5) Saves model + validation predictions
# 6) (Optional) Evaluates on Test_Data.csv if present
# ------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
import joblib
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import average_precision_score
# ---------- Paths ----------
TRAIN_CSV = r"C:\Users\cghiuta\Desktop\Training_Data.csv"
TEST_CSV  = r"C:\Users\cghiuta\Desktop\Testing_Data.csv"  # optional

# ---------- Load training data ----------
df = pd.read_csv(TRAIN_CSV)

# Target column (0=delivered, 1=undelivered/chargeback)
TARGET = "LABEL_UNDELIVERED_CB"
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found. Columns: {df.columns.tolist()}")

# Drop non-predictive identifiers if present
drop_cols = [c for c in ["MERCHANT_ID", "RN"] if c in df.columns]
df = df.drop(columns=drop_cols)

# ---------- Choose features ----------
# Numeric signals (use the ones that exist in your file)
numeric_candidates = [
    "TRUST_SCORE", "PRIOR_CB_RATE", "REFUND_RATE", "CANCEL_RATE",
    "WEBSITE_UPTIME", "SENTIMENT", "SALES_GROWTH_3M",
    "PAYOUT_DELAY_DAYS", "RESERVE_PERCENT", "DEPOSIT_POLICY_PERCENT",
    "DAYS_IN_ADVANCE", "BOOKING_AMOUNT", "BASE_FDR", "TYPICAL_HORIZON",
    "NEW_MERCHANT", "SHOCK_FLAG"
]
numeric_features = [c for c in numeric_candidates if c in df.columns]

# Categorical signals
categorical_candidates = ["VERTICAL", "COUNTRY", "MCC"]
categorical_features = [c for c in categorical_candidates if c in df.columns]

# Split X / y
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# ---------- Train/validation split (from your 80% training) ----------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# ---------- Preprocessing pipelines ----------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # with_mean=False keeps it compatible with sparse matrices downstream
    ("scaler", StandardScaler(with_mean=False))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

# ---------- Optuna Bayesian Optimization for Logistic Regression ----------
def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear"  # supports both l1 and l2

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver=solver,
        C=C,
        penalty=penalty
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    # Evaluate on validation (PR-AUC)
    y_prob_val = pipe.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, y_prob_val)
    return pr_auc

print("\n=== Starting Optuna Bayesian Optimization ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # try 30 trials; increase if time allows

print("\nBest trial:")
print(f"  PR-AUC: {study.best_trial.value:.4f}")
print(f"  Params: {study.best_trial.params}")

# ---------- Train final model with best params ----------
best_params = study.best_trial.params
final_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    C=best_params["C"],
    penalty=best_params["penalty"]
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", final_model)
])

pipe.fit(X_train, y_train)

# ---------- Evaluate on validation ----------
y_prob = pipe.predict_proba(X_valid)[:, 1]
y_pred = (y_prob >= 0.50).astype(int)  # default threshold 0.5

print("\n=== Validation — Threshold = 0.50 ===")
print(classification_report(y_valid, y_pred, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_valid, y_prob):.4f}")
print(f"PR-AUC (Average Precision): {average_precision_score(y_valid, y_prob):.4f}")
print("Confusion Matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_valid, y_pred))

# ---------- Optional: pick a better threshold ----------
# Many times you’ll want higher recall. Here’s a helper to print metrics for a chosen threshold.


def print_metrics_at_threshold(y_true, y_scores, thresh):
    yp = (y_scores >= thresh).astype(int)
    print(f"\n--- Metrics at threshold={thresh:.2f} ---")
    print(classification_report(y_true, yp, digits=4))
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, yp))

# Example: try 0.40 and 0.30 for more recall
print_metrics_at_threshold(y_valid, y_prob, 0.40)
print_metrics_at_threshold(y_valid, y_prob, 0.30)

# ---------- Save model and validation predictions ----------
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_path = OUT_DIR / "booking_risk_logreg_pipeline.pkl"
joblib.dump(pipe, model_path)
print(f"\nSaved model to: {model_path.resolve()}")

preds_df = X_valid.copy()
preds_df[TARGET] = y_valid.values
preds_df["pred_prob"] = y_prob
preds_df["pred_label_0p50"] = y_pred
preds_path = OUT_DIR / "validation_predictions_logreg.csv"
preds_df.to_csv(preds_path, index=False)
print(f"Saved validation predictions to: {preds_path.resolve()}")

# ---------- Optional: evaluate on Test_Data.csv if present ----------
if TEST_CSV is not None:
    test_df = pd.read_csv(TEST_CSV)
    # Make sure target exists; if not, just score probabilities
    has_label = TARGET in test_df.columns
    test_drop = [c for c in ["MERCHANT_ID", "RN"] if c in test_df.columns]
    test_df = test_df.drop(columns=test_drop)

    X_test = test_df.drop(columns=[TARGET]) if has_label else test_df
    y_test = test_df[TARGET].astype(int) if has_label else None

    y_test_prob = pipe.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.50).astype(int)

    test_preds = X_test.copy()
    if has_label:
        test_preds[TARGET] = y_test.values
    test_preds["pred_prob"] = y_test_prob
    test_preds["pred_label_0p50"] = y_test_pred
    test_out = OUT_DIR / "test_predictions_logreg.csv"
    test_preds.to_csv(test_out, index=False)
    print(f"\nSaved test predictions to: {test_out.resolve()}")

    if has_label:
        print("\n=== Test Set — Threshold = 0.50 ===")
        print(classification_report(y_test, y_test_pred, digits=4))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
        print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_test_prob):.4f}")
        print("Confusion Matrix [[TN FP] [FN TP]]:")
        print(confusion_matrix(y_test, y_test_pred))

# ---------- Optional: plot a Precision-Recall curve for validation ----------
try:
    precision, recall, thresh = precision_recall_curve(y_valid, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curve (Logistic Regression)")
    plt.tight_layout()
    pr_path = OUT_DIR / "validation_pr_curve.png"
    plt.savefig(pr_path, dpi=150)
    print(f"Saved PR curve to: {pr_path.resolve()}")
except Exception as e:
    print(f"(Optional PR plot skipped: {e})")
