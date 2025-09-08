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
    Sends df to FastAPI /score_batch and returns an enriched DataFrame
    with risk_score, risk_tier, suggested_reserve_percent, suggested_settlement_delay_days.
    """
    base = base_url.rstrip("/")
    results_all = []

    for i in range(0, len(df), chunk):
        sl = df.iloc[i:i+chunk]
        payload = _build_payload(sl, COL_MAP)

        items = payload["items"]  # <- extract the list
        r = requests.post(f"{base}/score_batch", json=items, timeout=120)
        r.raise_for_status()

        # Backend returns a list[ScoreOutWithContext]
        batch_results = r.json()
        results_all.extend(batch_results)

    # Convert response list to DataFrame (order preserved = input order)
    res_df = pd.DataFrame(results_all)

    # Build output by concatenating original df with response columns
    out = df.reset_index(drop=True).copy()
    # Align lengths as a sanity check
    if len(out) != len(res_df):
        raise RuntimeError(f"Response length {len(res_df)} != request length {len(out)}")

    wanted = [
        "risk_score",
        "risk_tier",
        "suggested_reserve_percent",
        "suggested_settlement_delay_days"
    ]
    for col in wanted:
        if col not in res_df.columns:
            raise RuntimeError(f"Backend response missing column: {col}")

    out["risk_score"] = res_df["risk_score"].astype(float)
    out["risk_tier"] = res_df["risk_tier"].astype(str)
    out["suggested_reserve_percent"] = res_df["suggested_reserve_percent"].astype(float)
    out["suggested_settlement_delay_days"] = res_df["suggested_settlement_delay_days"].astype(int)

    # convenience column for Watchlist/Business tab
    if "booking_amount" in out.columns:
        out["expected_loss$"] = out["risk_score"] * out["booking_amount"].astype(float)

    return out
