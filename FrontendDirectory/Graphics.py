# Streamlit frontend for PrePayGuard – Business-Oriented Policy Explorer
# ---------------------------------------------------------------------
# - Compares Actual Model Policy vs. Constant Risk Scenario for a selected merchant
# - KPIs a payments/risk buyer cares about (GMV, Expected Loss, Reserve $, Avg Delay)
# - Interactive Plotly charts for exploration + decision support
# - Uses POST /score_batch for fast scoring
#
# Requirements:
#   pip install streamlit plotly pandas requests
#
# How to run:
#   streamlit run Graphics.py
#
# Notes:
# - CSV must contain at least: MERCHANT_ID, VERTICAL, COUNTRY, DAYS_IN_ADVANCE, BOOKING_AMOUNT
# - If DEPOSIT_POLICY_PERCENT exists, it will be used to compute exposure; otherwise we assume full amount at risk.

import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(
    page_title="PrePayGuard – Business Policy Explorer",
    page_icon="💳",
    layout="wide"
)

TITLE = "PrePayGuard – Merchant Policy Explorer (Business View)"
st.title(TITLE)
st.caption("Predictive forward-delivery risk → dynamic reserves and settlement delays that protect you **without** over-penalizing low-risk sales.")

# -----------------------------
# Defaults (editable in sidebar)
# -----------------------------
BASE_DIR = Path(__file__).parent
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_CSV = "fdr_training_view_no_feature_engineering.csv"  # change if needed

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("1) Data & Backend")
api_base = st.sidebar.text_input("FastAPI base URL", value=str(DEFAULT_API_BASE))
score_batch_endpoint = f"{api_base.rstrip('/')}/score_batch"

csv_path_input = st.sidebar.text_input(
    "Bookings CSV path",
    value=DEFAULT_CSV
)

st.sidebar.header("2) Merchant & Sampling")
max_rows = st.sidebar.number_input("Max bookings to analyze", min_value=100, max_value=20000, value=2000, step=100)
const_prob = st.sidebar.slider("Constant scenario: risk score (p_fail)", 0.0, 1.0, 0.25, 0.01)

st.sidebar.header("3) Display Options")
color_by = st.sidebar.selectbox("Color by", options=["risk_tier", "vertical", "country"], index=0)
size_by_amount = st.sidebar.checkbox("Scale bubbles by booking amount", value=True)
horizon_bins = st.sidebar.selectbox("Horizon bucket size (days)", options=[7, 14, 30, 60], index=2)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Start backend with `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` and open http://localhost:8000/docs to verify `/score_batch`.")

# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["MERCHANT_ID","VERTICAL","COUNTRY","DAYS_IN_ADVANCE","BOOKING_AMOUNT"]

def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

@st.cache_data(show_spinner=False)
def load_bookings(csv_path: str) -> pd.DataFrame:
    path = resolve_path(csv_path)
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df

def call_score_batch(rows: pd.DataFrame) -> pd.DataFrame:
    payload = []
    for _, r in rows.iterrows():
        item = {
            "MERCHANT_ID": r["MERCHANT_ID"],
            "VERTICAL": r["VERTICAL"],
            "COUNTRY": r["COUNTRY"],
            "DAYS_IN_ADVANCE": float(r["DAYS_IN_ADVANCE"]),
            "BOOKING_AMOUNT": float(r["BOOKING_AMOUNT"]),
            # If your backend expects more INPUT_COLS, include them here (cast to float/int as needed).
            # Example (only if present in CSV):
            # "DEPOSIT_POLICY_PERCENT": float(r["DEPOSIT_POLICY_PERCENT"]) if "DEPOSIT_POLICY_PERCENT" in rows.columns else 0.0,
            # "TRUST_SCORE": float(r["TRUST_SCORE"]) if "TRUST_SCORE" in rows.columns else 70.0,
            # ...
        }
        payload.append(item)

    resp = requests.post(score_batch_endpoint, json=payload, timeout=60)
    if not resp.ok:
        # surface backend error details for quick fixes
        st.error(f"Batch scoring failed: {resp.status_code} {resp.reason}\n{resp.text}")
        st.stop()

    return pd.DataFrame(resp.json())

def compute_exposure(source_df: pd.DataFrame) -> pd.Series:
    """
    Business-friendly exposure proxy:
    - If DEPOSIT_POLICY_PERCENT exists, exposure = (1 - deposit%) * BOOKING_AMOUNT
    - Else, exposure = BOOKING_AMOUNT (conservative)
    """
    if "DEPOSIT_POLICY_PERCENT" in source_df.columns:
        dep = pd.to_numeric(source_df["DEPOSIT_POLICY_PERCENT"], errors="coerce").fillna(0.0) / 100.0
        exposure = (1.0 - dep).clip(0, 1) * pd.to_numeric(source_df.get("BOOKING_AMOUNT", 0.0), errors="coerce").fillna(0.0)
    else:
        exposure = pd.to_numeric(source_df.get("BOOKING_AMOUNT", 0.0), errors="coerce").fillna(0.0)
    return exposure

def local_policy_map(probability: float, days_in_advance: float):
    """
    Mirror of backend post_policy for the constant-risk scenario.
    Keep aligned with backend logic.
    """
    reserve = max(0.0, min(50.0, 100 * (0.08 * probability + 0.0009 * days_in_advance)))
    delay = int(max(0, min(45, int(15 * probability + 0.08 * days_in_advance))))
    if probability >= 0.60:
        tier = "high"
    elif probability >= 0.35:
        tier = "medium"
    else:
        tier = "low"
    return tier, round(reserve, 2), int(delay)

def kpi_block(title: str, value, delta=None, help_text=None):
    st.metric(label=title, value=value, delta=delta, help=help_text)

def format_currency(x):
    return f"${x:,.0f}"

# -----------------------------
# Load CSV & merchant selection
# -----------------------------
try:
    df_all = load_bookings(csv_path_input)
except Exception as e:
    st.error(f"Could not load bookings CSV: {e}")
    st.stop()

merchants = sorted(df_all["MERCHANT_ID"].unique().tolist())
sel_col1, sel_col2 = st.columns([2, 1])
with sel_col1:
    sel_merchant = st.selectbox("Select merchant", merchants, index=0)
with sel_col2:
    st.write("")

df_m = df_all[df_all["MERCHANT_ID"] == sel_merchant].copy()
if df_m.empty:
    st.warning("No bookings for selected merchant in the CSV.")
    st.stop()

df_m = df_m.head(int(max_rows)).copy()  # sample for speed

# -----------------------------
# Score via backend (Actual Model Policy)
# -----------------------------
st.subheader(f"Merchant: {sel_merchant} — Policy Comparison")
left_panel, right_panel = st.columns(2)

with st.spinner("Scoring bookings via /score_batch ..."):
    scored_df = call_score_batch(df_m)

# Ensure consistent types and presence
for needed in ["probability","suggested_reserve_percent","suggested_settlement_delay_days","risk_tier",
               "days_in_advance","booking_amount","vertical","country"]:
    if needed not in scored_df.columns:
        st.error(f"Backend response missing field: {needed}")
        st.stop()

# Attach exposure proxy from original df
scored_df["EXPOSURE"] = compute_exposure(df_m).values

# Constant-risk scenario for same bookings (mirrors policy locally)
const_df = df_m.copy()
const_df["probability"] = const_prob
tiers, reserves, delays = [], [], []
for _, r in const_df.iterrows():
    t, res, d = local_policy_map(float(const_prob), float(r["DAYS_IN_ADVANCE"]))
    tiers.append(t); reserves.append(res); delays.append(d)

const_df["risk_tier"] = tiers
const_df["suggested_reserve_percent"] = reserves
const_df["suggested_settlement_delay_days"] = delays
const_df.rename(columns={"DAYS_IN_ADVANCE":"days_in_advance","BOOKING_AMOUNT":"booking_amount",
                         "VERTICAL":"vertical","COUNTRY":"country"}, inplace=True)
const_df["EXPOSURE"] = compute_exposure(df_m).values

# -----------------------------
# KPIs a buyer understands
# -----------------------------
def summarize_kpis(df: pd.DataFrame):
    gmv = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0).sum()
    exp = pd.to_numeric(df["EXPOSURE"], errors="coerce").fillna(0)
    prob = pd.to_numeric(df["probability"], errors="coerce").fillna(0)
    el = (prob * exp).sum()  # Expected Loss proxy
    reserve_pct = pd.to_numeric(df["suggested_reserve_percent"], errors="coerce").fillna(0) / 100.0
    held = (reserve_pct * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0)).sum()
    avg_delay = pd.to_numeric(df["suggested_settlement_delay_days"], errors="coerce").fillna(0).mean()
    avg_prob = prob.mean()
    return {
        "GMV": gmv, "ExpectedLoss": el, "ReserveHeld": held,
        "AvgDelay": avg_delay, "AvgProb": avg_prob
    }

kpi_actual = summarize_kpis(scored_df)
kpi_const = summarize_kpis(const_df)

# KPI row – simple story for executives
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi_block("GMV Analyzed", format_currency(kpi_actual["GMV"]))
kpi_block("Expected Loss (Actual)", format_currency(kpi_actual["ExpectedLoss"]),
          delta=f"{format_currency(kpi_const['ExpectedLoss'] - kpi_actual['ExpectedLoss'])} vs Const")
kpi_block("Reserve Held (Actual)", format_currency(kpi_actual["ReserveHeld"]),
          delta=f"{format_currency(kpi_actual['ReserveHeld'] - kpi_const['ReserveHeld'])} vs Const")
kpi_block("Avg Settlement Delay (days)", f"{kpi_actual['AvgDelay']:.1f}",
          delta=f"{kpi_actual['AvgDelay']-kpi_const['AvgDelay']:+.1f} vs Const")

st.caption(
    "Interpretation: We want **lower Expected Loss** and **lower or right-sized Reserve Held**. "
    "Shorter average delay is better for merchant cash-flow, provided loss stays controlled."
)

# -----------------------------
# Charts – Business-first visuals
# -----------------------------

# 1) Frontier curves: mean Reserve% vs Days-in-Advance buckets — Actual vs Constant
def bucketize_days(df: pd.DataFrame, bucket: int) -> pd.DataFrame:
    b = int(bucket)
    d = df.copy()
    d["horizon_bucket"] = (pd.to_numeric(d["days_in_advance"], errors="coerce").fillna(0) // b) * b
    agg = d.groupby("horizon_bucket", as_index=False).agg(
        mean_reserve_pct=("suggested_reserve_percent", "mean"),
        gmv=("booking_amount", "sum"),
        n=("booking_amount", "count")
    )
    return agg

front_actual = bucketize_days(scored_df, horizon_bins)
front_const = bucketize_days(const_df, horizon_bins)

fig_frontier = go.Figure()
fig_frontier.add_trace(go.Scatter(
    x=front_actual["horizon_bucket"], y=front_actual["mean_reserve_pct"],
    mode="lines+markers", name="Actual Policy"
))
fig_frontier.add_trace(go.Scatter(
    x=front_const["horizon_bucket"], y=front_const["mean_reserve_pct"],
    mode="lines+markers", name=f"Constant p={const_prob:.2f}"
))
fig_frontier.update_layout(
    title="Policy Frontier: Average Reserve% by Service Horizon",
    xaxis_title=f"Days-in-Advance (bucketed by {horizon_bins})",
    yaxis_title="Average Reserve %",
    hovermode="x unified"
)

# 2) Tier distribution (where your exposure sits by tier)
def tier_bar(df: pd.DataFrame, title: str) -> go.Figure:
    tiers = df["risk_tier"].fillna("unknown")
    exposure = pd.to_numeric(df["EXPOSURE"], errors="coerce").fillna(0)
    # Sum exposure by tier (business likes exposure view, not just counts)
    tier_sums = (
        pd.DataFrame({"tier": tiers, "exposure": exposure})
        .groupby("tier", as_index=False)
        .sum().sort_values("exposure", ascending=False)
    )
    fig = px.bar(tier_sums, x="tier", y="exposure", text="exposure",
                 labels={"tier":"Risk Tier","exposure":"Exposure (approx $)"},
                 title=title)
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="Exposure (approx $)", xaxis_title="", uniformtext_minsize=10, uniformtext_mode='hide')
    return fig

fig_tier_actual = tier_bar(scored_df, "Exposure by Risk Tier (Actual Policy)")
fig_tier_const = tier_bar(const_df, "Exposure by Risk Tier (Constant Scenario)")

# 3) Bubble exploration (left/right panels): Days vs Reserve/Delay
def bubble_fig(df: pd.DataFrame, yfield: str, title: str):
    sz = "booking_amount" if size_by_amount else None
    color_field = color_by if color_by in df.columns else "risk_tier"
    fig = px.scatter(
        df, x="days_in_advance", y=yfield, size=sz, color=color_field,
        hover_data=["probability", "booking_amount", "vertical", "country", "suggested_settlement_delay_days"],
        labels={"days_in_advance":"Days in advance", yfield: ("Reserve %" if yfield=="suggested_reserve_percent" else "Delay (days)")},
        title=title
    )
    return fig

with left_panel:
    st.markdown("### Actual Model Policy (per booking)")
    st.plotly_chart(bubble_fig(scored_df, "suggested_reserve_percent", "Reserve% vs Horizon (Actual)"), use_container_width=True)
    st.plotly_chart(bubble_fig(scored_df, "suggested_settlement_delay_days", "Settlement Delay vs Horizon (Actual)"), use_container_width=True)

with right_panel:
    st.markdown(f"### Constant Risk Scenario (p={const_prob:.2f})")
    st.plotly_chart(bubble_fig(const_df, "suggested_reserve_percent", "Reserve% vs Horizon (Constant)"), use_container_width=True)
    st.plotly_chart(bubble_fig(const_df, "suggested_settlement_delay_days", "Settlement Delay vs Horizon (Constant)"), use_container_width=True)

# 4) ROI view: Reserve Held vs Expected Loss — Actual vs Constant
roi_fig = go.Figure()
roi_fig.add_trace(go.Bar(
    name="Expected Loss (Actual)", x=["Actual"], y=[kpi_actual["ExpectedLoss"]]
))
roi_fig.add_trace(go.Bar(
    name="Expected Loss (Constant) ", x=["Constant"], y=[kpi_const["ExpectedLoss"]]
))
roi_fig.add_trace(go.Bar(
    name="Reserve Held (Actual)", x=["Actual"], y=[kpi_actual["ReserveHeld"]]
))
roi_fig.add_trace(go.Bar(
    name="Reserve Held (Constant) ", x=["Constant"], y=[kpi_const["ReserveHeld"]]
))
roi_fig.update_layout(
    title="ROI View: Expected Loss vs Reserve Held (lower is better)",
    barmode="group",
    yaxis_title="$"
)

st.markdown("### Policy Frontiers & Tier Mix")
front_cols = st.columns(2)
with front_cols[0]:
    st.plotly_chart(fig_frontier, use_container_width=True)
with front_cols[1]:
    tabs = st.tabs(["Actual", "Constant"])
    with tabs[0]:
        st.plotly_chart(fig_tier_actual, use_container_width=True)
    with tabs[1]:
        st.plotly_chart(fig_tier_const, use_container_width=True)

# 5) Top exposure at-risk table (business loves a short actionable list)
st.markdown("### Top Exposure — Drilldown")
rank_df = scored_df.copy()
rank_df["ExposureAtRisk"] = (pd.to_numeric(rank_df["probability"], errors="coerce").fillna(0)
                             * pd.to_numeric(rank_df["EXPOSURE"], errors="coerce").fillna(0))
topn = rank_df.sort_values("ExposureAtRisk", ascending=False).head(20)
st.dataframe(
    topn[["vertical","country","days_in_advance","booking_amount","probability",
          "suggested_reserve_percent","suggested_settlement_delay_days","risk_tier","ExposureAtRisk"]],
    use_container_width=True
)

# Download scored results
dl_cols = st.columns(2)
with dl_cols[0]:
    st.download_button(
        label="Download ACTUAL policy results (CSV)",
        data=scored_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sel_merchant}_actual_policy.csv",
        mime="text/csv",
    )
with dl_cols[1]:
    st.download_button(
        label="Download CONSTANT scenario results (CSV)",
        data=const_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sel_merchant}_constant_{const_prob:.2f}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "This view helps risk & payments teams tune reserves/delays by horizon and tier. "
    "Drag/hover to explore; compare Expected Loss vs. Reserve Held to justify policy choices to merchants and schemes."
)
