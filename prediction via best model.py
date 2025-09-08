# predict_one.py
# Minimal script: load model, prep one custom row, predict.

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "xgb_best_model.pkl"
TRAIN_CSV  = "fdr_training_view_no_feature_engineering.csv"
TARGET_COL = "label_undelivered_cb"
CATEGORICALS = ["vertical", "country"]   # adjust if you added/changed categoricals
DROP_COLS = ["merchant_id"]              # adjust if you dropped other columns during training
THRESHOLD = 0.5

def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_training_schema():
    df = pd.read_csv(TRAIN_CSV)
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {TRAIN_CSV}")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df, feature_cols

def fit_encoders(df):
    encoders = {}
    for col in CATEGORICALS:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            encoders[col] = le
    return encoders

def apply_encoders(row_df, encoders):
    # Expect categorical columns to be object at this point
    for col, le in encoders.items():
        if col not in row_df.columns:
            continue
        val = row_df.at[row_df.index[0], col]

        # Accept already-encoded ints in range
        if isinstance(val, (int, np.integer)) and 0 <= val < len(le.classes_):
            continue

        if pd.isna(val):
            # If missing, you may choose a default like the most frequent class (index 0)
            row_df.at[row_df.index[0], col] = 0
            continue

        # Otherwise encode string label
        label = str(val)
        if label not in set(le.classes_):
            raise ValueError(
                f"Unseen category '{label}' for '{col}'. "
                f"Known examples: {list(le.classes_)[:10]}{'...' if len(le.classes_)>10 else ''}"
            )
        row_df.at[row_df.index[0], col] = int(le.transform([label])[0])

    # Cast encoded categoricals to int to avoid mixed dtypes downstream
    for col in encoders.keys():
        if col in row_df.columns:
            row_df[col] = pd.to_numeric(row_df[col], errors="raise").astype("int64")

    return row_df

def build_row(custom_merchant, feature_cols, model, encoders):
    # Initialize with proper dtypes:
    # - object for categoricals (to safely hold strings before encoding)
    # - float for everything else (more permissive than int)
    init_data = {}
    for c in feature_cols:
        if c in encoders:  # categorical
            init_data[c] = pd.Series([None], dtype="object")
        else:
            init_data[c] = pd.Series([np.nan], dtype="float64")
    row = pd.DataFrame(init_data)

    # Fill provided values, coercing numerics where appropriate
    for k, v in custom_merchant.items():
        if k not in row.columns:
            raise KeyError(f"Provided key '{k}' not in training features.\nExpected: {feature_cols}")

        if k in encoders:
            # leave as string/int; encoder will handle it
            row.at[0, k] = v
        else:
            # numeric-ish: coerce to numeric
            row.at[0, k] = pd.to_numeric(v, errors="coerce")

    # Encode categoricals and cast to int
    row = apply_encoders(row, encoders)

    # Ensure numeric columns are numeric and fill any NaNs
    non_cat = [c for c in feature_cols if c not in encoders]
    if non_cat:
        row[non_cat] = row[non_cat].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Align to model input order if available
    if hasattr(model, "feature_names_in_"):
        missing = [c for c in model.feature_names_in_ if c not in row.columns]
        if missing:
            raise ValueError(f"Missing features required by model: {missing}")
        row = row[list(model.feature_names_in_)]
    else:
        row = row[feature_cols]

    return row

def predict_one(custom_merchant):
    model = load_model()
    df_train, feature_cols = load_training_schema()
    encoders = fit_encoders(df_train)

    # Warn if you forgot some features
    missing = [c for c in feature_cols if c not in custom_merchant]
    if missing:
        print(f"[NOTE] You did not provide {len(missing)} feature(s). They will default to 0.")
        print(f"       Missing (first 20): {missing[:20]}")

    X_row = build_row(custom_merchant, feature_cols, model, encoders)

    # probability (positive class) + binary prediction
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_row)[:, 1][0])
    else:
        # for rare cases (e.g., booster) without predict_proba
        prob = float(model.predict(X_row)[0])
    label = int(prob >= THRESHOLD)
    return prob, label

if __name__ == "__main__":
    # EXAMPLE: replace with the real merchant values
    custom_merchant = {
    "vertical": "subscription_box",
    "mcc": 5968,
    "country": "SG",
    "age_months": 60,
    "new_merchant": 0.0289,
    "trust_score": 69,
    "prior_cb_rate": 0.034,
    "refund_rate": 0.0708,
    "cancel_rate": 0.1328,
    "website_uptime": 0.9847,
    "sentiment": 0.0775,
    "sales_growth_3m": 0.0497,
    "payout_delay_days": 5.5359,
    "reserve_percent": 7.11,
    "deposit_policy_percent": 10.31,
    "days_in_advance": 49,
    "booking_amount": 69.73,
    "shock_flag": 0,
    "typical_horizon": 49,
    "base_fdr": 0.1958,
}

    prob, label = predict_one(custom_merchant)
    print(f"Predicted probability (label_undelivered_cb=1): {prob:.4f}")
    print(f"Predicted label (threshold={THRESHOLD}): {label}")
