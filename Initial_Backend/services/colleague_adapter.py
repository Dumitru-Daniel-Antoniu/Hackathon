# services/colleague_adapter.py
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def _lower_and_alias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _ensure_numeric(s, default=0.0):
    if s is None:
        return None
    if s.dtype == "O":
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
        s = s.replace({"": None, "nan": None, "None": None})
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(default)

def _labelize(series: pd.Series, known_classes: list[str] | None, fallback_token: str = "other"):
    ser = series.astype(str).fillna(fallback_token)
    if known_classes:
        classes = list(known_classes)
        if fallback_token not in classes:
            classes.append(fallback_token)
        classes = np.array(classes, dtype=object)
        lut = {c: i for i, c in enumerate(classes)}
        idx = ser.str.lower().map(lambda v: lut.get(v, lut[fallback_token]))
        return idx.astype(int), classes
    else:
        le = LabelEncoder()
        le.fit(ser)
        return pd.Series(le.transform(ser), index=ser.index).astype(int), le.classes_

def _prep_features_df(df_raw: pd.DataFrame,
                      feature_list: list[str] | None,
                      classes: dict | None) -> pd.DataFrame:
    df = _lower_and_alias(df_raw)

    # Coerce numerics broadly
    for col in [
        "trust_score","prior_cb_rate","refund_rate","cancel_rate","sentiment","sales_growth_3m",
        "payout_delay_days","reserve_percent","deposit_policy_percent","days_in_advance",
        "booking_amount","new_merchant","shock_flag","base_fdr","typical_horizon",
        "age_months","website_uptime","mcc"
    ]:
        if col in df.columns:
            df[col] = _ensure_numeric(df[col], 0.0)

    # Cats
    vert_known = (classes or {}).get("vertical")
    ctry_known = (classes or {}).get("country")
    if "vertical" in df.columns:
        df["vertical"], _ = _labelize(df["vertical"].astype(str).str.lower(), vert_known)
    else:
        df["vertical"] = 0
    if "country" in df.columns:
        df["country"], _ = _labelize(df["country"].astype(str).str.upper(), ctry_known)
    else:
        df["country"] = 0

    # Drop clear non-features if present
    for dropc in ["merchant_id","label_undelivered_cb"]:
        if dropc in df.columns:
            df = df.drop(columns=[dropc])

    # If we have a feature_list from training, enforce that order (add missing as zeros)
    if feature_list:
        for c in feature_list:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feature_list]
    else:
        # otherwise leave as-is (later we’ll align to booster expectations)
        df = df.copy()

    # Ensure float32
    for c in df.columns:
        df[c] = _ensure_numeric(df[c], 0.0).astype(np.float32)
    return df

def _get_booster(model):
    return model.get_booster() if hasattr(model, "get_booster") else model

def predict_proba_colleague(model,
                            df_raw: pd.DataFrame,
                            feature_list: list[str] | None,
                            classes: dict | None) -> np.ndarray:
    dfF = _prep_features_df(df_raw, feature_list, classes)
    booster = _get_booster(model)

    # Read booster feature metadata if available
    try:
        train_names = booster.feature_names  # list[str] or None
    except Exception:
        train_names = None
    try:
        nfeat = booster.num_features()
    except Exception:
        nfeat = None

    # Case 1: booster has feature names -> build a DataFrame with exact names/order
    if train_names:
        # Create frame with exactly training names; fill missing with 0, drop extras
        X_df = pd.DataFrame({name: (dfF[name] if name in dfF.columns else 0.0) for name in train_names})
        dtest = xgb.DMatrix(X_df)  # pandas -> carries names
        preds = booster.predict(dtest, validate_features=True)

    else:
        # Case 2: booster has no names -> use raw NumPy, ensure correct column count
        X = dfF.to_numpy(dtype=np.float32, copy=False)
        if nfeat is not None:
            if X.shape[1] > nfeat:
                X = X[:, :nfeat]
            elif X.shape[1] < nfeat:
                pad = np.zeros((X.shape[0], nfeat - X.shape[1]), dtype=np.float32)
                X = np.hstack([X, pad])
        dtest = xgb.DMatrix(X)  # no names
        # Disable validation against names (there are none)
        try:
            preds = booster.predict(dtest, validate_features=False)
        except TypeError:
            preds = booster.predict(dtest)

    # Ensure 1D probs
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 2:
        preds = preds[:, 1]
    return preds.astype(float)
