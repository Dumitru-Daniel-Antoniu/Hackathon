import streamlit as st
import pandas as pd
from pathlib import Path
from utilities.api import score_via_api

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(1100px 600px at 10% -10%, #0b1220 0%, #0b1220 38%, #0e1a2b 60%, #0f2338 80%, #0f273f 100%);
        color: #e7edf5;
        overflow: auto;
      }
      .block-container { padding-top: 4.5rem; padding-bottom: 0.8rem; }

      :root { --accent: #19c6d1; --accent-2: #7ae2f2; }
      h1, h2, h3 { letter-spacing: .2px; }

      .hero {
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
      }

      .subtle { color: #cfe3f2; opacity: .85; font-size: 0.98rem; line-height: 1.6; }
      .glow { text-shadow: 0 0 24px rgba(26,198,209,.35); }

      .svg-card{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: .8rem;
        height: 100%;
      }
      .img-caption { margin-top: .5rem; }

      /* Adjusted spacers */
      .spacer-lg { height: 1.6rem; }
      .spacer-md { height: 0.8rem; } /* use this before Why section */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📥 Upload & Data Check")
st.caption(
    "Drop a CSV. If it’s raw, we’ll score it automatically. "
    "If it’s already scored, we’ll use it as-is. "
    "If no file is uploaded, we’ll use a built-in demo dataset."
)

# --- Sidebar: backend URL (shared across pages) ---
api_base = st.sidebar.text_input(
    "FastAPI base URL",
    value=st.session_state.get("api_base", "http://localhost:8000"),
)
st.session_state["api_base"] = api_base

# --- Local default CSV (used when nothing is uploaded) ---
FRONTEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = FRONTEND_DIR / "default_data" / "Nonscored_bookings.csv"

# --- Defaults to fill if the CSV is raw and lacks model inputs ---
DEFAULTS = {
    "mcc": 5999,
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
    "airline": 4511,
    "online_travel_agency": 4722,
    "tour_operator": 4722,
    "event_ticketing": 7922,
    "hotel": 7011,
    "education_courses": 8299,
    "transport_shuttle": 4121,
    "digital_services": 5817,
    "subscription_box": 5968,
    "retail_goods_prepay": 5964,
}

# Minimum columns we need for business views (and to join back context)
MIN_REQUIRED = ["merchant_id", "vertical", "country", "days_in_advance", "booking_amount"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _ensure_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the DataFrame has everything the backend expects (all lowercase).
    Does *not* touch any probability fields.
    """
    df = _normalize(df)

    # Core minimum for business views
    missing = [c for c in MIN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Optional MCC derivation from vertical (string names)
    if "mcc" not in df.columns and "vertical" in df.columns:
        df["mcc"] = (
            df["vertical"].astype(str).str.lower().map(VERTICAL_TO_MCC).fillna(DEFAULTS["mcc"]).astype(int)
        )

    # Fill other defaults if absent
    for k, v in DEFAULTS.items():
        if k not in df.columns:
            df[k] = v

    # Basic typing for numeric inputs expected by the model
    int_cols = ["mcc", "payout_delay_days", "days_in_advance", "new_merchant", "shock_flag"]
    float_cols = [
        "trust_score",
        "prior_cb_rate",
        "refund_rate",
        "cancel_rate",
        "website_uptime",
        "sentiment",
        "sales_growth_3m",
        "reserve_percent",
        "deposit_policy_percent",
        "booking_amount",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    return df


def _normalize_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create/normalize `risk_probability` in [0,1] and `risk_score_pct` in [0,100].
    Accepts backends/CSVs that may have:
      - 'risk_probability' already (0-1 or 0-100)
      - 'risk_score' (0-1 or 0-100)
      - 'probability' (0-1 or 0-100)
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def _coerce_to_unit_interval(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        # If values look like percentages (>1), convert to 0-1 first
        s = s.where(s <= 1.0, s / 100.0)
        # Finally, clip to [0,1] to avoid any stray >1 values
        return s.clip(0, 1).fillna(0.0)

    if "risk_probability" in cols:
        df["risk_probability"] = _coerce_to_unit_interval(df[cols["risk_probability"]])
    elif "risk_score" in cols:
        df["risk_probability"] = _coerce_to_unit_interval(df[cols["risk_score"]])
    elif "probability" in cols:
        df["risk_probability"] = _coerce_to_unit_interval(df[cols["probability"]])
    else:
        # If none provided, create an empty column; subsequent pages will still work
        df["risk_probability"] = 0.0

    df["risk_score_pct"] = (df["risk_probability"] * 100).round(2)
    return df


def _is_already_scored(df: pd.DataFrame) -> bool:
    low = {c.lower() for c in df.columns}
    return any(c in low for c in ("risk_probability", "risk_score", "probability"))


# --- Source selection: upload or default dataset ---
uploaded = st.file_uploader("Upload bookings CSV", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source_label = f"uploaded file ({len(raw):,} rows)"
else:
    if DEFAULT_DATA.exists():
        raw = pd.read_csv(DEFAULT_DATA)
        source_label = f"default dataset: {DEFAULT_DATA.name} ({len(raw):,} rows)"
        st.info("Using the built-in demo dataset until you upload your own.")
    else:
        st.warning("No file uploaded and the default dataset is missing. Please upload a CSV.")
        st.stop()

raw = _normalize(raw)

# --- Score if needed; otherwise normalize scoring fields ---
with st.spinner("Processing…"):
    if _is_already_scored(raw):
        # File already contains probabilities/scores; normalize and go
        scored = _normalize_probs(raw)
        st.success("Detected a scored file — no API call needed.")
    else:
        # Build a model-ready payload, call backend, then *normalize the response too*
        to_score = _ensure_inputs(raw)
        scored = score_via_api(to_score, st.session_state["api_base"])
        scored = _normalize_probs(scored)
        st.success("Scored successfully via backend.")

# --- Business calc: expected loss (risk_probability * booking_amount) ---
if "expected_loss$" not in scored.columns:
    scored["expected_loss$"] = (
        pd.to_numeric(scored["risk_probability"], errors="coerce").fillna(0.0)
        * pd.to_numeric(scored["booking_amount"], errors="coerce").fillna(0.0)
    )

# Persist for other tabs
st.session_state["scored_df"] = scored

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(scored))
c2.metric("GMV analyzed", f"${pd.to_numeric(scored['booking_amount'], errors='coerce').fillna(0.0).sum():,.0f}")
c3.metric("Avg chance of non-delivery", f"{pd.to_numeric(scored['risk_probability'], errors='coerce').fillna(0.0).mean():.2%}")
c4.metric("Expected loss (approx)", f"${pd.to_numeric(scored['expected_loss$'], errors='coerce').fillna(0.0).sum():,.0f}")

st.caption(f"Source: **{source_label}**")
st.dataframe(scored.head(300), use_container_width=True)

st.info("Proceed to one of the pages from the sidebar. Your data is cached for all of them.")
