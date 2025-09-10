# utilities/api.py
import pandas as pd
import requests
from typing import Dict, List

# CSV -> Backend schema mapping (left = your CSV col, right = backend field)
COL_MAP: Dict[str, str] = {
    "merchant_id": "MERCHANT_ID",
    "mcc": "MCC",
    "vertical": "VERTICAL",
    "country": "COUNTRY",

    "trust_score": "TRUST_SCORE",
    "prior_cb_rate": "PRIOR_CB_RATE",
    "refund_rate": "REFUND_RATE",
    "cancel_rate": "CANCEL_RATE",
    "sentiment": "SENTIMENT",
    "sales_growth_3m": "SALES_GROWTH_3M",

    "payout_delay_days": "PAYOUT_DELAY_DAYS",
    "reserve_percent": "RESERVE_PERCENT",
    "deposit_policy_percent": "DEPOSIT_POLICY_PERCENT",

    "days_in_advance": "DAYS_IN_ADVANCE",
    "booking_amount": "BOOKING_AMOUNT",
    "new_merchant": "NEW_MERCHANT",
    "shock_flag": "SHOCK_FLAG",
}

# Required numeric fields the backend validates strongly
_NUMERIC_FLOAT = [
    "TRUST_SCORE","PRIOR_CB_RATE","REFUND_RATE","CANCEL_RATE","SENTIMENT",
    "SALES_GROWTH_3M","PAYOUT_DELAY_DAYS","RESERVE_PERCENT","DEPOSIT_POLICY_PERCENT",
    "DAYS_IN_ADVANCE","BOOKING_AMOUNT"
]
_NUMERIC_INT = ["NEW_MERCHANT","SHOCK_FLAG"]
# Optional context fields (can be absent/NaN and will be dropped from payload)
_OPTIONAL = ["MERCHANT_ID","MCC","VERTICAL","COUNTRY"]


def _build_payload(df: pd.DataFrame, col_map: Dict[str, str]) -> dict:
    missing = [c for c in col_map.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.reset_index(drop=True).copy()
    backend_cols = {col_map[k]: k for k in col_map.keys()}

    items: List[dict] = []
    for _, row in df.iterrows():
        item = {}
        for be_name, csv_name in backend_cols.items():
            val = row[csv_name]

            # Skip optional fields if NaN/empty
            if be_name in _OPTIONAL and (pd.isna(val) or val == ""):
                continue

            # Type normalization
            if be_name in _NUMERIC_INT:
                val = int(val)
            elif be_name in _NUMERIC_FLOAT:
                val = float(val)

            item[be_name] = val

        items.append(item)

    return {"items": items}


def _to_unit_interval(series: pd.Series) -> pd.Series:
    """Convert numeric to [0,1]; if values look like percentages (>1), divide by 100."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s <= 1.0, s / 100.0)
    return s.clip(0, 1).fillna(0.0)


def score_via_api(df: pd.DataFrame, base_url: str, chunk: int = 3000) -> pd.DataFrame:
    """
    Send df to FastAPI /score/batch and return an enriched DataFrame.
    Produces:
      - risk_probability (0..1)
      - risk_score_pct (0..100)
      - risk_tier, suggested_reserve_percent, suggested_settlement_delay_days
      - expected_loss$
    """
    base = base_url.rstrip("/")
    results_all: List[dict] = []

    for i in range(0, len(df), chunk):
        sl = df.iloc[i:i + chunk]
        payload = _build_payload(sl, COL_MAP)
        r = requests.post(f"{base}/score/batch", json=payload["items"], timeout=120)
        r.raise_for_status()
        results_all.extend(r.json())

    res_df = pd.DataFrame(results_all)
    out = df.reset_index(drop=True).copy()
    if len(out) != len(res_df):
        raise RuntimeError(f"Response length {len(res_df)} != request length {len(out)}")

    # ---- Normalize probability (0..1) with correct precedence ----
    cols = set(res_df.columns.str.lower())
    if "risk_score" in cols:
        p = _to_unit_interval(res_df[[c for c in res_df.columns if c.lower() == "risk_score"][0]])
    elif "probability" in cols:
        p = _to_unit_interval(res_df[[c for c in res_df.columns if c.lower() == "probability"][0]])
    elif "risk_probability" in cols:
        p = _to_unit_interval(res_df[[c for c in res_df.columns if c.lower() == "risk_probability"][0]])
    else:
        raise RuntimeError("Backend response missing 'risk_score'/'probability'/'risk_probability'")

    out["risk_probability"] = p.astype(float)
    out["risk_score_pct"] = (out["risk_probability"] * 100).round(2)

    # ---- Copy other fields (with basic sanitation) ----
    if any(c.lower() == "risk_tier" for c in res_df.columns):
        out["risk_tier"] = res_df[[c for c in res_df.columns if c.lower() == "risk_tier"][0]].astype(str)

    if any(c.lower() == "suggested_reserve_percent" for c in res_df.columns):
        out["suggested_reserve_percent"] = pd.to_numeric(
            res_df[[c for c in res_df.columns if c.lower() == "suggested_reserve_percent"][0]],
            errors="coerce"
        ).fillna(0.0)

    if any(c.lower() == "suggested_settlement_delay_days" for c in res_df.columns):
        out["suggested_settlement_delay_days"] = pd.to_numeric(
            res_df[[c for c in res_df.columns if c.lower() == "suggested_settlement_delay_days"][0]],
            errors="coerce"
        ).fillna(0).astype(int)

    # ---- Business calc: expected loss uses probability (not %!) ----
    if "booking_amount" in out.columns:
        out["expected_loss$"] = (
            out["risk_probability"] * pd.to_numeric(out["booking_amount"], errors="coerce").fillna(0.0)
        )

    return out
