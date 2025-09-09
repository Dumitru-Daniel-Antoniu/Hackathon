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
    # Check columns exist
    missing = [c for c in col_map.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.reset_index(drop=True).copy()

    # Remap to backend names (keep a copy of original df for merging later)
    backend_cols = {col_map[k]: k for k in col_map.keys()}

    items: List[dict] = []
    for _, row in df.iterrows():
        item = {}
        # Fill mapped fields
        for be_name, csv_name in backend_cols.items():
            val = row[csv_name]

            # Skip optional fields if NaN/None
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

def score_via_api(df: pd.DataFrame, base_url: str, chunk: int = 3000) -> pd.DataFrame:
    """
    Send df to FastAPI /score/batch and return an enriched DataFrame.
    Computes with `probability` (0..1). Also exposes `risk_score_pct` for display.
    """
    base = base_url.rstrip("/")
    results_all = []

    for i in range(0, len(df), chunk):
        sl = df.iloc[i:i + chunk]
        payload = _build_payload(sl, COL_MAP)

        items = payload["items"]  # list to POST
        r = requests.post(f"{base}/score/batch", json=items, timeout=120)
        r.raise_for_status()

        batch_results = r.json()  # list[ScoreOutWithContext]
        results_all.extend(batch_results)

    # Convert response to DataFrame (order preserved)
    res_df = pd.DataFrame(results_all)

    out = df.reset_index(drop=True).copy()
    if len(out) != len(res_df):
        raise RuntimeError(f"Response length {len(res_df)} != request length {len(out)}")

    # ---- Normalize probability (0..1) regardless of how backend sends it ----
    if "probability" in res_df.columns:
        p = pd.to_numeric(res_df["probability"], errors="coerce").clip(0, 1).fillna(0.0)
    elif "risk_score" in res_df.columns:
        # risk_score may be 0..1 (legacy) or 0..100 (percent). Normalize to 0..1
        rs = pd.to_numeric(res_df["risk_score"], errors="coerce").fillna(0.0)
        p = rs.where(rs <= 1.0, rs / 100.0).clip(0, 1)
    else:
        raise RuntimeError("Backend response missing 'probability'/'risk_score'")

    # ---- Copy other fields (with basic sanitation) ----
    out["risk_probability"] = p
    out["risk_score_pct"] = (p * 100).round(2)  # nice for display if you still want a % column

    if "risk_tier" in res_df.columns:
        out["risk_tier"] = res_df["risk_tier"].astype(str)
    if "suggested_reserve_percent" in res_df.columns:
        out["suggested_reserve_percent"] = pd.to_numeric(
            res_df["suggested_reserve_percent"], errors="coerce"
        ).fillna(0.0)
    if "suggested_settlement_delay_days" in res_df.columns:
        out["suggested_settlement_delay_days"] = pd.to_numeric(
            res_df["suggested_settlement_delay_days"], errors="coerce"
        ).fillna(0).astype(int)

    # ---- Business calc: expected loss uses probability (not %!) ----
    if "booking_amount" in out.columns:
        out["expected_loss$"] = (
            out["risk_probability"] * pd.to_numeric(out["booking_amount"], errors="coerce").fillna(0.0)
        )

    return out

