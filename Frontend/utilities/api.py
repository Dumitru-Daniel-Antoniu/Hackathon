# utilities/api.py
import io

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
    "TRUST_SCORE", "PRIOR_CB_RATE", "REFUND_RATE", "CANCEL_RATE", "SENTIMENT",
    "SALES_GROWTH_3M", "PAYOUT_DELAY_DAYS", "RESERVE_PERCENT", "DEPOSIT_POLICY_PERCENT",
    "DAYS_IN_ADVANCE", "BOOKING_AMOUNT"
]
_NUMERIC_INT = ["NEW_MERCHANT", "SHOCK_FLAG"]
# Optional context fields (can be absent/NaN and will be dropped from payload)
_OPTIONAL = ["MERCHANT_ID", "MCC", "VERTICAL", "COUNTRY"]


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


def score_via_api(df: pd.DataFrame, base_url: str) -> pd.DataFrame:
    """
    Send df once to FastAPI /score/csv (multipart/form-data) and return an enriched DataFrame.
    Produces (and keeps backward-compat):
      - risk_probability (0..1)
      - risk_score_pct (0..100)
      - risk_score  (alias of risk_probability)   <-- for legacy UI
      - probability (alias of risk_probability)   <-- for legacy UI
      - risk_tier, suggested_reserve_percent, suggested_settlement_delay_days
      - expected_loss$
    """
    base = base_url.rstrip("/")

    # --- prepare CSV to match backend headers ---
    sl = df.copy().rename(columns=COL_MAP)
    keep = [c for c in COL_MAP.values() if c in sl.columns]
    sl = sl[keep]

    # coerce numerics so CSV cells are clean
    for c in _NUMERIC_FLOAT:
        if c in sl.columns:
            sl[c] = pd.to_numeric(sl[c], errors="coerce")
    for c in _NUMERIC_INT:
        if c in sl.columns:
            sl[c] = pd.to_numeric(sl[c], errors="coerce").fillna(0).astype("Int64")

    buf = io.StringIO()
    sl.to_csv(buf, index=False, lineterminator="\n")
    files = {"file": ("batch.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    # --- single call to CSV endpoint ---
    r = requests.post(f"{base}/score/csv", files=files, timeout=120)
    r.raise_for_status()

    # Backend returns List[ScoreOutWithContext]
    res_df = pd.DataFrame(r.json())

    out = df.reset_index(drop=True).copy()
    if len(out) != len(res_df):
        raise RuntimeError(f"Response length {len(res_df)} != request length {len(out)}")

    # ---- Normalize probability (0..1) with correct precedence ----
    cols_lower = [c.lower() for c in res_df.columns]
    if "risk_score" in cols_lower:
        pcol = res_df.columns[cols_lower.index("risk_score")]
    elif "probability" in cols_lower:
        pcol = res_df.columns[cols_lower.index("probability")]
    elif "risk_probability" in cols_lower:
        pcol = res_df.columns[cols_lower.index("risk_probability")]
    else:
        raise RuntimeError("Backend response missing 'risk_score'/'probability'/'risk_probability'")

    p = pd.to_numeric(res_df[pcol], errors="coerce")
    # if values look like percentages (>1), treat as % and convert to 0..1
    p = p.where(p <= 1.0, p / 100.0).clip(0, 1).fillna(0.0)

    # core + backward-compat columns
    out["risk_probability"] = p.astype(float)
    out["risk_score_pct"] = (out["risk_probability"] * 100).round(2)
    out["risk_score"] = out["risk_probability"]  # legacy alias
    out["probability"] = out["risk_probability"]  # legacy alias

    # ---- Copy other fields (with basic sanitation) ----
    if "risk_tier" in cols_lower:
        out["risk_tier"] = res_df[res_df.columns[cols_lower.index("risk_tier")]].astype(str)

    if "suggested_reserve_percent" in cols_lower:
        col = res_df.columns[cols_lower.index("suggested_reserve_percent")]
        out["suggested_reserve_percent"] = pd.to_numeric(res_df[col], errors="coerce").fillna(0.0)

    if "suggested_settlement_delay_days" in cols_lower:
        col = res_df.columns[cols_lower.index("suggested_settlement_delay_days")]
        out["suggested_settlement_delay_days"] = pd.to_numeric(res_df[col], errors="coerce").fillna(0).astype(int)

    # ---- Business calc: expected loss uses probability (not %!) ----
    if "booking_amount" in out.columns:
        out["expected_loss$"] = (
                out["risk_probability"] * pd.to_numeric(out["booking_amount"], errors="coerce").fillna(0.0)
        )

    return out
