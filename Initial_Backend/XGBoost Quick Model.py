# train_booking_risk_xgb.py
# ------------------------------------------------------------
# 1) Load Training_Data.csv (+ optional Test_Data.csv)
# 2) Add engineered features:
#    - days_in_advance_bucket (categorical)
#    - refund_cancel_ratio (numeric)
#    - high_risk_vertical_flag (0/1 from MCC or VERTICAL)
#    - shock_adjusted_lead (numeric interaction)
#    - merchant_stability (numeric composite)
# 3) Preprocess (impute + one-hot for categoricals)
# 4) Bayesian optimization (Optuna) to tune XGBoost hyperparameters
# 5) Train XGBoost (class imbalance handled via scale_pos_weight)
# 6) Evaluate (val + optional test), save predictions & plots
# 7) Save feature importance and model artifacts
# ------------------------------------------------------------

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import optuna
from xgboost import XGBClassifier

# ---------- Paths ----------
TRAIN_CSV = r"C:\Users\cghiuta\Desktop\Training_Data.csv"
TEST_CSV  = r"C:\Users\cghiuta\Desktop\Testing_Data.csv"  # optional
OUT_DIR   = Path("artifacts_xgb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "LABEL_UNDELIVERED_CB"

# ---------- Engineered features helper ----------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Lead time bucket (categorical)
    if "DAYS_IN_ADVANCE" in df.columns:
        df["days_in_advance_bucket"] = pd.cut(
            df["DAYS_IN_ADVANCE"],
            bins=[-1, 7, 30, 90, 180, np.inf],
            labels=["0-7", "8-30", "31-90", "91-180", "180+"],
            include_lowest=True
        )
    else:
        df["days_in_advance_bucket"] = "unknown"

    # 2) Refund/Cancel ratio (numeric)
    if {"REFUND_RATE", "CANCEL_RATE"}.issubset(df.columns):
        denom = (df["CANCEL_RATE"].astype(float).replace(0, np.nan)).fillna(1e-6)
        df["refund_cancel_ratio"] = df["REFUND_RATE"].astype(float) / denom
    else:
        df["refund_cancel_ratio"] = 0.0

    # 3) High-risk vertical flag (0/1) via MCC if present, else via VERTICAL
    if "MCC" in df.columns:
        mcc_num = pd.to_numeric(df["MCC"], errors="coerce")
        df["high_risk_vertical_flag"] = mcc_num.isin([4511, 4722, 7922]).astype(int)
    elif "VERTICAL" in df.columns:
        high_vert = {"airline", "tour_operator", "event_ticketing", "events", "ticketing"}
        df["high_risk_vertical_flag"] = (
            df["VERTICAL"].astype(str).str.lower().isin(high_vert)
        ).astype(int)
    else:
        df["high_risk_vertical_flag"] = 0

    # 4) Shock-adjusted lead time (numeric interaction)
    if {"DAYS_IN_ADVANCE", "SHOCK_FLAG"}.issubset(df.columns):
        df["shock_adjusted_lead"] = df["DAYS_IN_ADVANCE"].astype(float) * df["SHOCK_FLAG"].astype(float)
    else:
        df["shock_adjusted_lead"] = 0.0

    # 5) Merchant stability index (numeric composite)
    if {"TRUST_SCORE", "REFUND_RATE", "CANCEL_RATE"}.issubset(df.columns):
        df["merchant_stability"] = (
            df["TRUST_SCORE"].astype(float)
            - (df["REFUND_RATE"].astype(float).fillna(0) + df["CANCEL_RATE"].astype(float).fillna(0)) * 100.0
        )
    else:
        df["merchant_stability"] = 0.0

    return df

# ---------- Load training data ----------
df = pd.read_csv(TRAIN_CSV)
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found. Columns: {df.columns.tolist()}")

# Drop non-predictive identifiers if present
df = df.drop(columns=[c for c in ["MERCHANT_ID", "RN"] if c in df.columns], errors="ignore")

# Add engineered features
df = add_engineered_features(df)

# ---------- Select features ----------
# Base numeric/categorical
base_numeric = [
    "TRUST_SCORE", "PRIOR_CB_RATE", "REFUND_RATE", "CANCEL_RATE",
    "SENTIMENT", "SALES_GROWTH_3M",
    "PAYOUT_DELAY_DAYS", "RESERVE_PERCENT", "DEPOSIT_POLICY_PERCENT",
    "DAYS_IN_ADVANCE", "BOOKING_AMOUNT",
    "NEW_MERCHANT", "SHOCK_FLAG","BASE_FDR","TYPICAL_HORIZON"
]
base_categorical = ["VERTICAL", "COUNTRY"]

# Engineered
eng_numeric = ["refund_cancel_ratio", "shock_adjusted_lead", "merchant_stability", "high_risk_vertical_flag"]
eng_categorical = ["days_in_advance_bucket"]

numeric_features = [c for c in base_numeric + eng_numeric if c in df.columns]
categorical_features = [c for c in base_categorical + eng_categorical if c in df.columns]

X = df[numeric_features + categorical_features].copy()
y = df[TARGET].astype(int).values

# ---------- Train/Validation split ----------
X_train_df, X_valid_df, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# ---------- Preprocess (impute + one-hot for categoricals) ----------
numeric_transformer = SimpleImputer(strategy="median")
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

# Fit on train; transform train/valid
X_train = preprocess.fit_transform(X_train_df)
X_valid = preprocess.transform(X_valid_df)
feature_names = preprocess.get_feature_names_out()

# ---------- Handle class imbalance ----------
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = max(1.0, neg / max(1, pos))  # >1 if positives are rarer

# ---------- XGBoost + Bayesian Optimization (Optuna) ----------
def objective(trial):
    params = {
        "n_estimators": 5000,  # generous upper bound; early stopping will trim it
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 55,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    y_val_prob = model.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, y_val_prob)
    return pr_auc

print("\n=== Starting Optuna Bayesian Optimization (XGBoost) ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)  # increase to 80–120 if you have more time

print("\nBest trial:")
print(f"  PR-AUC: {study.best_trial.value:.4f}")
print(f"  Params: {study.best_trial.params}")

# Train final model with best params (and early stopping again)
best_params = study.best_trial.params
best_params.update({
    "n_estimators": 5000,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 55,
    "n_jobs": -1,
    "scale_pos_weight": scale_pos_weight,
})
xgb = XGBClassifier(**best_params)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

best_iter = getattr(xgb, "best_iteration", None) or xgb.n_estimators
print(f"Best iteration: {best_iter}")

# ---------- Evaluate on validation ----------
y_prob = xgb.predict_proba(X_valid)[:, 1]
y_pred = (y_prob >= 0.50).astype(int)

def dump_metrics(prefix, y_true, y_scores, y_labels):
    rep = classification_report(y_true, y_labels, digits=4, output_dict=True)
    cm  = confusion_matrix(y_true, y_labels)
    roc = roc_auc_score(y_true, y_scores)
    ap  = average_precision_score(y_true, y_scores)
    print(f"\n=== {prefix} ===")
    print(classification_report(y_true, y_labels, digits=4))
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC (Average Precision): {ap:.4f}")
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(cm)
    return {
        "prefix": prefix,
        "accuracy": rep["accuracy"],
        "precision_pos": rep["1"]["precision"],
        "recall_pos": rep["1"]["recall"],
        "f1_pos": rep["1"]["f1-score"],
        "roc_auc": roc,
        "pr_auc": ap,
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])
    }

metrics = []
metrics.append(dump_metrics("Validation@0.50", y_valid, y_prob, y_pred))
for thr in [0.40, 0.30]:
    y_pred_t = (y_prob >= thr).astype(int)
    metrics.append(dump_metrics(f"Validation@{thr:.2f}", y_valid, y_prob, y_pred_t))

pd.DataFrame(metrics).to_csv(OUT_DIR / "xgb_validation_metrics.csv", index=False)

# ---------- Save predictions ----------
preds_df = X_valid_df.copy()
preds_df[TARGET] = y_valid
preds_df["pred_prob"] = y_prob
preds_df["pred_label_0p50"] = y_pred
preds_df.to_csv(OUT_DIR / "xgb_validation_predictions.csv", index=False)

# ---------- Plots: ROC + PR ----------
fpr, tpr, _ = roc_curve(y_valid, y_prob)
precision, recall, _ = precision_recall_curve(y_valid, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Validation ROC Curve (XGBoost)")
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_validation_roc.png", dpi=150)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation Precision-Recall Curve (XGBoost)")
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_validation_pr.png", dpi=150)

# ---------- Feature importance (mapped to names) ----------
score_dict = xgb.get_booster().get_score(importance_type="gain")
mapped_importance = {}
for i, fname in enumerate(feature_names):
    key = f"f{i}"
    mapped_importance[fname] = score_dict.get(key, 0.0)

fi = pd.DataFrame(list(mapped_importance.items()), columns=["feature", "importance_gain"]) \
       .sort_values("importance_gain", ascending=False)
fi.to_csv(OUT_DIR / "xgb_feature_importance_gain.csv", index=False)

topN = fi.head(20)
plt.figure(figsize=(8, 6))
plt.barh(topN["feature"], topN["importance_gain"])
plt.gca().invert_yaxis()
plt.xlabel("Gain")
plt.title("Top 20 Features by Gain (XGBoost)")
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_feature_importance_top20.png", dpi=150)

# ---------- Save model + preprocessing ----------
joblib.dump(preprocess, OUT_DIR / "preprocess_ohe.pkl")
joblib.dump(xgb, OUT_DIR / "booking_risk_xgb_model.pkl")
print(f"\nSaved model and artifacts to: {OUT_DIR.resolve()}")

# ---------- Optional: Evaluate on Test_Data.csv if present ----------
if TEST_CSV and os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    test_df = test_df.drop(columns=[c for c in ["MERCHANT_ID", "RN"] if c in test_df.columns], errors="ignore")
    test_df = add_engineered_features(test_df)

    has_label = TARGET in test_df.columns
    if has_label:
        y_test = test_df[TARGET].astype(int).values
        X_test_df = test_df[numeric_features + categorical_features]
    else:
        X_test_df = test_df[numeric_features + categorical_features]

    X_test = preprocess.transform(X_test_df)
    y_test_prob = xgb.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.50).astype(int)

    out = X_test_df.copy()
    out["pred_prob"] = y_test_prob
    out["pred_label_0p50"] = y_test_pred
    if has_label:
        out[TARGET] = y_test
    out.to_csv(OUT_DIR / "xgb_test_predictions.csv", index=False)

    if has_label:
        test_metrics = dump_metrics("Test@0.50", y_test, y_test_prob, y_test_pred)
        pd.DataFrame([test_metrics]).to_csv(OUT_DIR / "xgb_test_metrics.csv", index=False)
