import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Lead time bucket
    if "DAYS_IN_ADVANCE" in df.columns:
        df["days_in_advance_bucket"] = pd.cut(
            df["DAYS_IN_ADVANCE"],
            bins=[-1, 7, 30, 90, 180, np.inf],
            labels=["0-7","8-30","31-90","91-180","180+"],
            include_lowest=True
        ).astype("object")
    else:
        df["days_in_advance_bucket"] = "unknown"

    # 2) Refund/Cancel ratio
    if {"REFUND_RATE","CANCEL_RATE"}.issubset(df.columns):
        denom = (df["CANCEL_RATE"].astype(float).replace(0, np.nan)).fillna(1e-6)
        df["refund_cancel_ratio"] = df["REFUND_RATE"].astype(float) / denom
    else:
        df["refund_cancel_ratio"] = 0.0

    # 3) High-risk vertical flag (MCC optional; VERTICAL fallback)
    if "MCC" in df.columns:
        mcc_num = pd.to_numeric(df["MCC"], errors="coerce")
        df["high_risk_vertical_flag"] = mcc_num.isin([4511, 4722, 7922]).astype(int)
    else:
        high_vert = {"airline","tour_operator","event_ticketing","events","ticketing"}
        if "VERTICAL" in df.columns:
            df["high_risk_vertical_flag"] = (
                df["VERTICAL"].astype(str).str.lower().isin(high_vert).astype(int)
            )
        else:
            df["high_risk_vertical_flag"] = 0

    # 4) Shock-adjusted lead (NOTE: using (1 - shock) is often more intuitive; keep your version if intended)
    if {"DAYS_IN_ADVANCE","SHOCK_FLAG"}.issubset(df.columns):
        df["shock_adjusted_lead"] = (
            df["DAYS_IN_ADVANCE"].astype(float) * df["SHOCK_FLAG"].astype(float)
        )
    else:
        df["shock_adjusted_lead"] = 0.0

    # 5) Merchant stability
    if {"TRUST_SCORE","REFUND_RATE","CANCEL_RATE"}.issubset(df.columns):
        df["merchant_stability"] = (
            df["TRUST_SCORE"].astype(float)
            - (df["REFUND_RATE"].astype(float).fillna(0) + df["CANCEL_RATE"].astype(float).fillna(0)) * 100.0
        )
    else:
        df["merchant_stability"] = 0.0

    return df
