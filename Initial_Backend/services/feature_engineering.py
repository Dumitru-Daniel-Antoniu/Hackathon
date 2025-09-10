import numpy as np
import pandas as pd

def _first_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name that exists (case-sensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _ensure_numeric(s, default=0.0):
    """Coerce a Series to numeric (handle strings, commas, blanks)."""
    if s is None:
        return None
    if s.dtype == "O":
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
        s = s.replace({"": None, "nan": None, "None": None})
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(default)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Accept lowercase or uppercase payloads
    # (Keep existing names; only read with aliases below)
    # Example aliases you may read from:
    col_TRUST_SCORE         = _first_col(df, "TRUST_SCORE", "trust_score")
    col_PRIOR_CB_RATE       = _first_col(df, "PRIOR_CB_RATE", "prior_cb_rate")
    col_REFUND_RATE         = _first_col(df, "REFUND_RATE", "refund_rate")
    col_CANCEL_RATE         = _first_col(df, "CANCEL_RATE", "cancel_rate")
    col_SENTIMENT           = _first_col(df, "SENTIMENT", "sentiment")
    col_SALES_GROWTH_3M     = _first_col(df, "SALES_GROWTH_3M", "sales_growth_3m")
    col_PAYOUT_DELAY_DAYS   = _first_col(df, "PAYOUT_DELAY_DAYS", "payout_delay_days")
    col_RESERVE_PERCENT     = _first_col(df, "RESERVE_PERCENT", "reserve_percent")
    col_DEPOSIT_POLICY_PCT  = _first_col(df, "DEPOSIT_POLICY_PERCENT", "deposit_policy_percent")
    col_DAYS_IN_ADVANCE     = _first_col(df, "DAYS_IN_ADVANCE", "days_in_advance")
    col_BOOKING_AMOUNT      = _first_col(df, "BOOKING_AMOUNT", "booking_amount")
    col_NEW_MERCHANT        = _first_col(df, "NEW_MERCHANT", "new_merchant")
    col_SHOCK_FLAG          = _first_col(df, "SHOCK_FLAG", "shock_flag")
    col_MCC                 = _first_col(df, "MCC", "mcc")
    col_VERTICAL            = _first_col(df, "VERTICAL", "vertical")

    # ---- Coerce numeric columns (handles CSV strings)
    for col in [
        col_TRUST_SCORE, col_PRIOR_CB_RATE, col_REFUND_RATE, col_CANCEL_RATE,
        col_SENTIMENT, col_SALES_GROWTH_3M, col_PAYOUT_DELAY_DAYS, col_RESERVE_PERCENT,
        col_DEPOSIT_POLICY_PCT, col_DAYS_IN_ADVANCE, col_BOOKING_AMOUNT,
        col_NEW_MERCHANT, col_SHOCK_FLAG
    ]:
        if col is not None and col in df.columns:
            df[col] = _ensure_numeric(df[col], default=0.0)

    # 1) Lead time bucket (SAFE after coercion)
    if col_DAYS_IN_ADVANCE is not None:
        try:
            df["days_in_advance_bucket"] = pd.cut(
                df[col_DAYS_IN_ADVANCE],
                bins=[-1, 7, 30, 90, 180, np.inf],
                labels=["0-7","8-30","31-90","91-180","180+"],
                include_lowest=True
            ).astype("object")
        except Exception:
            # Fallback: unknown bucket if something odd slips through
            df["days_in_advance_bucket"] = "unknown"
    else:
        df["days_in_advance_bucket"] = "unknown"

    # 2) Refund/Cancel ratio
    if col_REFUND_RATE and col_CANCEL_RATE:
        denom = df[col_CANCEL_RATE].replace(0, np.nan).fillna(1e-6)
        df["refund_cancel_ratio"] = df[col_REFUND_RATE] / denom
    else:
        df["refund_cancel_ratio"] = 0.0

    # 3) High-risk vertical flag (by MCC if present, else by VERTICAL)
    if col_MCC and col_MCC in df.columns:
        mcc_num = _ensure_numeric(df[col_MCC], default=np.nan)
        df["high_risk_vertical_flag"] = mcc_num.isin([4511, 4722, 7922]).astype(int)
    else:
        high_vert = {"airline","tour_operator","event_ticketing","events","ticketing"}
        if col_VERTICAL and col_VERTICAL in df.columns:
            df["high_risk_vertical_flag"] = (
                df[col_VERTICAL].astype(str).str.lower().isin(high_vert).astype(int)
            )
        else:
            df["high_risk_vertical_flag"] = 0

    # 4) Shock-adjusted lead
    if col_DAYS_IN_ADVANCE and col_SHOCK_FLAG:
        df["shock_adjusted_lead"] = df[col_DAYS_IN_ADVANCE] * df[col_SHOCK_FLAG]
    else:
        df["shock_adjusted_lead"] = 0.0

    # 5) Merchant stability
    if col_TRUST_SCORE and (col_REFUND_RATE or col_CANCEL_RATE):
        r = df[col_REFUND_RATE] if col_REFUND_RATE else 0.0
        c = df[col_CANCEL_RATE] if col_CANCEL_RATE else 0.0
        df["merchant_stability"] = df[col_TRUST_SCORE] - (pd.Series(r).fillna(0) + pd.Series(c).fillna(0)) * 100.0
    else:
        df["merchant_stability"] = 0.0

    return df
