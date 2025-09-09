# Frontend/pages/Upload_and_Data_Check.py
import streamlit as st
import pandas as pd
from utilities.api import score_via_api

st.title("📥 Upload & Data Check")
st.caption("Drop a CSV. If it’s raw, we’ll score it automatically. If it’s already scored, we’ll use it as-is.")

# --- Sidebar: backend URL (shared) ---
api_base = st.sidebar.text_input("FastAPI base URL", value=st.session_state.get("api_base","http://localhost:8000"))
st.session_state["api_base"] = api_base

# --- Defaults to fill if the CSV is raw and lacks model inputs ---
DEFAULTS = {
    "mcc": 5999,                          # generic retail fallback
    "trust_score": 65.0,
    "prior_cb_rate": 0.01,
    "refund_rate": 0.05,
    "cancel_rate": 0.05,
    "website_uptime": 0.985,
    "sentiment": 0.0,
    "sales_growth_3m": 0.0,
    "payout_delay_days": 0,
    "reserve_percent": 0.0,
    "deposit_policy_percent": 0.0,
    "new_merchant": 0,
    "shock_flag": 0,
}

VERTICAL_TO_MCC = {
    "airline": 4511, "online_travel_agency": 4722, "tour_operator": 4722,
    "event_ticketing": 7922, "hotel": 7011, "education_courses": 8299,
    "transport_shuttle": 4121, "digital_services": 5817,
    "subscription_box": 5968, "retail_goods_prepay": 5964
}

MIN_REQUIRED = ["merchant_id","vertical","country","days_in_advance","booking_amount"]

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _ensure_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the DataFrame has everything the backend expects (lowercase)."""
    df = df.copy()

    # Core minimum for business views
    missing = [c for c in MIN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Optional MCC derivation from vertical
    if "mcc" not in df.columns and "vertical" in df.columns:
        df["mcc"] = df["vertical"].astype(str).str.lower().map(VERTICAL_TO_MCC).fillna(DEFAULTS["mcc"]).astype(int)

    # Fill other defaults if absent
    for k, v in DEFAULTS.items():
        if k not in df.columns:
            df[k] = v

    # Basic typing
    int_cols = ["mcc","payout_delay_days","days_in_advance","new_merchant","shock_flag"]
    float_cols = [
        "trust_score","prior_cb_rate","refund_rate","cancel_rate","website_uptime","sentiment",
        "sales_growth_3m","reserve_percent","deposit_policy_percent","booking_amount"
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    return df

def _is_scored(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    return {"risk_score", "suggested_reserve_percent", "suggested_settlement_delay_days"}.issubset(cols)

uploaded = st.file_uploader("Upload bookings CSV", type=["csv"])

if not uploaded:
    st.info("Tip: you can upload either raw or already-scored CSV.")
    st.stop()

raw = pd.read_csv(uploaded)
raw = _normalize(raw)

with st.spinner("Processing…"):
    if _is_scored(raw):
        scored = raw.copy()
        st.success("Detected an already-scored file — no API call needed.")
    else:
        # ensure all inputs then call your backend
        to_score = _ensure_inputs(raw)
        scored = score_via_api(to_score, st.session_state["api_base"])
        st.success("Scored successfully via backend.")

# Make sure we have expected_loss$ for downstream pages
if "expected_loss$" not in scored.columns:
    scored["expected_loss$"] = pd.to_numeric(scored.get("risk_score", scored.get("probability", 0.0)), errors="coerce").fillna(0.0) \
                               * pd.to_numeric(scored["booking_amount"], errors="coerce").fillna(0.0)

# Persist for other tabs
st.session_state["scored_df"] = scored

# Quick KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(scored))
c2.metric("GMV analyzed", f"${scored['booking_amount'].sum():,.0f}")
c3.metric("Avg chance of non-delivery", f"{scored.get('risk_score', scored.get('probability', 0.0)).mean():.2%}")
c4.metric("Expected loss (approx)", f"${scored['expected_loss$'].sum():,.0f}")

st.dataframe(scored.head(300), use_container_width=True)

# CTA
st.info("Proceed to **Executive Summary** from the sidebar. Your data is cached for all pages.")
