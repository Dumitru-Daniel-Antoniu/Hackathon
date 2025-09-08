# app.py
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts_xgb"))
PREPROCESS_PATH = ARTIFACT_DIR / "preprocess_ohe.pkl"
MODEL_PATH = ARTIFACT_DIR / "booking_risk_xgb_model.pkl"

# ---------- Load artifacts once ----------
preprocess = joblib.load(PREPROCESS_PATH)   # sklearn ColumnTransformer (fitted)
model = joblib.load(MODEL_PATH)             # XGBClassifier (fitted)

# Columns the preprocess expects as INPUT (before OHE/transform)
# (sklearn>=1.0 keeps this after fit; else define manually as in training)
try:
    INPUT_COLS = list(preprocess.feature_names_in_)
except Exception:
    # fallback: define explicitly to match your training script
    INPUT_COLS = [
        # numeric base
        "TRUST_SCORE","PRIOR_CB_RATE","REFUND_RATE","CANCEL_RATE",
        "SENTIMENT","SALES_GROWTH_3M","PAYOUT_DELAY_DAYS","RESERVE_PERCENT",
        "DEPOSIT_POLICY_PERCENT","DAYS_IN_ADVANCE","BOOKING_AMOUNT",
        "NEW_MERCHANT","SHOCK_FLAG",
        # engineered numeric
        "refund_cancel_ratio","shock_adjusted_lead","merchant_stability","high_risk_vertical_flag",
        # categorical base + engineered
        "VERTICAL","COUNTRY","days_in_advance_bucket"
    ]

# ---------- Feature engineering (same as training) ----------
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
        mcc_num = pd.to_numeric(df["MCC"], errors="ignore")
        df["high_risk_vertical_flag"] = (mcc_num.isin([4511,4722,7922]).astype(int)
                                         if hasattr(mcc_num, "isin") else 0)
    else:
        high_vert = {"airline","tour_operator","event_ticketing","events","ticketing"}
        if "VERTICAL" in df.columns:
            df["high_risk_vertical_flag"] = df["VERTICAL"].astype(str).str.lower().isin(high_vert).astype(int)
        else:
            df["high_risk_vertical_flag"] = 0

    # 4) Shock-adjusted lead
    if {"DAYS_IN_ADVANCE","SHOCK_FLAG"}.issubset(df.columns):
        df["shock_adjusted_lead"] = df["DAYS_IN_ADVANCE"].astype(float) * df["SHOCK_FLAG"].astype(float)
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

# ---------- Simple policy (example) ----------
def post_policy(prob: float, days_ahead: float):
    # map prob → risk tier
    if prob >= 0.41:
        tier = "Fraudulent Actor"
    elif prob >= 0.21:
        tier = "High-Risk Counterparty"
    elif prob >= 0.11:
        tier = "Developing Organization"
    elif prob >= 0.06:
        tier = "Established Operator"
    else:
        tier = "Trusted Partner"
    # simple knobs (you can tune these)
    reserve_percent = float(np.clip(100*(0.08*prob + 0.0009*days_ahead), 0, 50))
    settlement_delay = int(np.clip(15*prob + 0.08*days_ahead, 0, 45))
    return tier, reserve_percent, settlement_delay

# ---------- Request/Response schemas ----------
class BookingIn(BaseModel):
    # minimal set — add others if you collect them in your UI
    MERCHANT_ID: Optional[str] = None
    MCC: Optional[int] = None
    VERTICAL: Optional[str] = None
    COUNTRY: Optional[str] = None

    # TRUST_SCORE: Optional[float]
    # PRIOR_CB_RATE: Optional[float]
    # REFUND_RATE: Optional[float]
    # CANCEL_RATE: Optional[float]
    # SENTIMENT: Optional[float]
    # SALES_GROWTH_3M: Optional[float]
    #
    # PAYOUT_DELAY_DAYS: Optional[float]
    # RESERVE_PERCENT: Optional[float]
    # DEPOSIT_POLICY_PERCENT: Optional[float]

    DAYS_IN_ADVANCE: float
    BOOKING_AMOUNT: float
    # NEW_MERCHANT: Optional[int] = Field(..., description="0 or 1")
    # SHOCK_FLAG: Optional[int] = Field(..., description="0 or 1")

class ScoreOut(BaseModel):
    probability: float
    risk_score: float
    risk_tier: Literal["Trusted Partner","Established Operator","Developing Organization","High-Risk Counterparty","Fraudulent Actor"]
    suggested_reserve_percent: float
    suggested_settlement_delay_days: int

class ScoreOutWithContext(ScoreOut):
    merchant_id: Optional[str] = None
    vertical: Optional[str] = None
    country: Optional[str] = None
    days_in_advance: Optional[float] = None
    booking_amount: Optional[float] = None

app = FastAPI(title="FDR Booking Risk Scorer", version="1.0")

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Single scoring ----------
@app.post("/score", response_model=ScoreOutWithContext)
def score_one(b: BookingIn):
    raw = pd.DataFrame([b.model_dump()])
    raw_fe = add_engineered_features(raw)

    # Ensure expected columns exist; fill missing with NA
    for col in INPUT_COLS:
        if col not in raw_fe.columns:
            raw_fe[col] = np.nan

    X_in = raw_fe[INPUT_COLS]
    X_tr = preprocess.transform(X_in)
    prob = float(model.predict_proba(X_tr)[:, 1][0])

    tier, reserve, delay = post_policy(prob, float(b.DAYS_IN_ADVANCE))

    return ScoreOutWithContext(
        probability=prob,
        risk_score=prob,
        risk_tier=tier,
        suggested_reserve_percent=round(reserve,2),
        suggested_settlement_delay_days=delay,
        merchant_id=b.MERCHANT_ID,
        vertical=b.VERTICAL,
        country=b.COUNTRY,
        days_in_advance=b.DAYS_IN_ADVANCE,
        booking_amount=b.BOOKING_AMOUNT
    )

# ---------- Batch scoring ----------
@app.post("/score_batch", response_model=List[ScoreOutWithContext])
def score_batch(items: List[BookingIn]):
    df = pd.DataFrame([it.dict() for it in items])
    df_fe = add_engineered_features(df)

    for col in INPUT_COLS:
        if col not in df_fe.columns:
            df_fe[col] = np.nan
    X_in = df_fe[INPUT_COLS]
    X_tr = preprocess.transform(X_in)
    probs = model.predict_proba(X_tr)[:, 1]

    results = []
    for i, p in enumerate(probs):
        b = items[i]
        tier, reserve, delay = post_policy(float(p), float(b.DAYS_IN_ADVANCE))
        results.append(ScoreOutWithContext(
            probability=float(p),
            risk_score=float(p),
            risk_tier=tier,
            suggested_reserve_percent=round(reserve,2),
            suggested_settlement_delay_days=delay,
            merchant_id=b.MERCHANT_ID,
            vertical=b.VERTICAL,
            country=b.COUNTRY,
            days_in_advance=b.DAYS_IN_ADVANCE,
            booking_amount=b.BOOKING_AMOUNT
        ))
    return results
