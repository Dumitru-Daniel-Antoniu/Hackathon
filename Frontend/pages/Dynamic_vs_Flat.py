import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

PAGE_TITLE = "💡 Dynamic vs Flat — Executive View"
PAGE_CAPTION = (
    "Compare a flat reserve policy (same % for every merchant) with the model‑driven "
    "dynamic policy (different % based on risk). See how much loss you still carry "
    "and how much cash you need to hold back under each approach."
)

st.set_page_config(page_title="Dynamic vs Flat Executive", layout="wide")
st.title(PAGE_TITLE)
st.caption(PAGE_CAPTION)

# ---------------------------------------------------------------------------
# 0) Get the scored dataset prepared on the Upload page
# ---------------------------------------------------------------------------
if "scored_df" not in st.session_state:
    st.warning(
        "No scored data found. Go to **Upload & Data Check** first, upload a CSV, and score it. "
        "Then come back to this page."
    )
    st.stop()

df = st.session_state["scored_df"].copy()
df.columns = [c.strip().lower() for c in df.columns]

# Robust probability accessor (handles risk_probability / risk_score / probability)
def _prob_series(d: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in d.columns}
    if "risk_probability" in cols:
        s = pd.to_numeric(d[cols["risk_probability"]], errors="coerce")
    elif "risk_score" in cols:
        s = pd.to_numeric(d[cols["risk_score"]], errors="coerce")
    elif "probability" in cols:
        s = pd.to_numeric(d[cols["probability"]], errors="coerce")
        s = s.where(s <= 1.0, s / 100.0)  # convert 0–100 -> 0–1
    else:
        s = pd.Series(0.0, index=d.index)
    return s.clip(0, 1).fillna(0.0)

# Fallback dynamic policy if column is missing (mirrors backend formula)
def _dynamic_percent_fallback(p: pd.Series, days: pd.Series) -> pd.Series:
    # reserve% = 100*(0.08*p + 0.0009*days), clipped to [0,50]
    r = 100.0 * (0.08 * p + 0.0009 * pd.to_numeric(days, errors="coerce").fillna(0.0))
    return r.clip(lower=0.0, upper=50.0)

p = _prob_series(df)
amount = pd.to_numeric(df.get("booking_amount", 0.0), errors="coerce").fillna(0.0)

# Dynamic %: use model output if available; otherwise fallback
if "suggested_reserve_percent" in df.columns:
    dyn_pct = pd.to_numeric(df["suggested_reserve_percent"], errors="coerce").fillna(0.0)
else:
    dyn_pct = _dynamic_percent_fallback(p, df.get("days_in_advance", 0))

if "deposit_policy_percent" in df.columns:
    dep = pd.to_numeric(df["deposit_policy_percent"], errors="coerce").fillna(0.0) / 100.0
    exposure = (1.0 - dep).clip(0, 1) * amount
else:
    exposure = amount

# Portfolio baselines
GMV = float(amount.sum())
EL = float((p * exposure).sum())  # Expected Loss with no reserves

# ---------------------------------------------------------------------------
# 1) Controls (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.subheader("Policy you want to compare")
flat_pct = st.sidebar.slider(
    "Flat reserve % (same for every merchant)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.1,
    format="%.1f",
    help="This is the single percentage you hold back from every payout, regardless of risk."
)

# ---------------------------------------------------------------------------
# 2) Helper to compute portfolio metrics for a given reserve% vector
# ---------------------------------------------------------------------------
def portfolio_metrics(reserve_pct: pd.Series) -> dict:
    reserve_pct = pd.to_numeric(reserve_pct, errors="coerce").fillna(0.0)
    # Dollar reserves held (cap at booking amount to avoid >100%)
    reserves = (reserve_pct / 100.0 * amount).clip(upper=amount)
    total_reserves = float(reserves.sum())

    # For the simple exec view we mirror your doc logic:
    # Net loss after reserves = EL - total_reserves (floored at 0)
    loss_absorbed = float(min(total_reserves, EL))
    net_loss = float(max(EL - total_reserves, 0.0))

    return {
        "gmv": GMV,
        "expected_loss": EL,
        "reserves_held": total_reserves,
        "loss_absorbed": loss_absorbed,
        "net_loss": net_loss,
        "pct_volume_withheld": (total_reserves / GMV) if GMV > 0 else 0.0,
        "pct_loss_absorbed": (loss_absorbed / EL) if EL > 0 else 0.0,
    }

flat_metrics = portfolio_metrics(pd.Series(flat_pct, index=df.index))
dyn_metrics = portfolio_metrics(dyn_pct)

# Headline improvement (how much less loss with Dynamic vs Flat)
savings_vs_flat = flat_metrics["net_loss"] - dyn_metrics["net_loss"]

def money(x: float) -> str:
    return f"${x:,.0f}"

def pct(x: float) -> str:
    return f"{100*x:,.1f}%"

# ---------------------------------------------------------------------------
# 3) Context header for execs
# ---------------------------------------------------------------------------
st.subheader("What this page shows")
st.caption(
    "Left: a **Flat** policy that withholds the same % from every merchant. "
    "Right: a **Dynamic** policy that withholds more where risk is higher and less where it’s lower. "
)

# Portfolio top line
t1, t2 = st.columns(2)
with t1:
    st.metric("Bookings total sum (GMV)", money(GMV))
with t2:
    st.metric("Expected loss with no reserves", money(EL))


st.divider()
# ---------------------------------------------------------------------------
# 4) Split view: Flat (left) vs Dynamic (right)
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("### Flat policy")
    st.caption(
        f"You set a single percentage for everyone. The slider is currently **{flat_pct}%**. "
    )
    st.metric("Reserves held", money(flat_metrics["reserves_held"]),
              help="How much cash you would withhold from payouts under this flat %.")
    st.metric("Net loss after reserves", money(flat_metrics["net_loss"]),
              help="What you still expect to lose after applying reserves.")
    st.metric("% of loss absorbed", pct(flat_metrics["pct_loss_absorbed"]))
    st.metric("% of volume withheld", pct(flat_metrics["pct_volume_withheld"]))

with right:
    st.markdown("### Dynamic policy")
    st.caption(
        "The model recommends a different reserve % per booking, based on risk. "
    )
    st.metric("Reserves held", money(dyn_metrics["reserves_held"]))
    st.metric("Net loss after reserves", money(dyn_metrics["net_loss"]))
    st.metric("% of loss absorbed", pct(dyn_metrics["pct_loss_absorbed"]))
    st.metric("% of volume withheld", pct(dyn_metrics["pct_volume_withheld"]))

# ---------------------------------------------------------------------------
# 5) Callout: business impact vs Flat
# ---------------------------------------------------------------------------
if savings_vs_flat > 0:
    st.success(
        f"**Money saved with Dynamic vs Flat {flat_pct}%:** {money(savings_vs_flat)} **gained** using Dynamic Policy "
        f"({pct((savings_vs_flat / EL) if EL > 0 else 0)} of total expected loss)."
    )
elif savings_vs_flat < 0:
    st.warning(
        f"Dynamic would result in **{money(-savings_vs_flat)} more** expected loss than Flat {flat_pct}% "
        f"(at these parameters)."
    )
else:
    st.info("Dynamic and Flat result in the same expected loss at the current setting.")

# “Cash vs. Risk Left” — grouped bars
policies = ["Flat", "Dynamic"]
recovered = [flat_metrics["loss_absorbed"], dyn_metrics["loss_absorbed"]]

# Optional: per-$ efficiency (how many $ of loss are covered per $1 reserved)
efficiency = []
for m in (flat_metrics, dyn_metrics):
    eff = (m["loss_absorbed"] / m["reserves_held"]) if m["reserves_held"] > 0 else 0.0
    efficiency.append(eff)

fig1 = go.Figure(go.Bar(
    name="Loss absorbed (recovered)",
    x=policies,
    y=recovered,
    text=[money(v) for v in recovered],           # "$" labels on top of bars
    textposition="outside",
    customdata=efficiency,                         # attach efficiency for hover
    hovertemplate=(
        "%{x}<br>"
        "Money regained: %{y:$,.0f}<br>"
        "<extra></extra>"
    )
))

fig1.update_layout(
    title="How much money was recovered according to each policy. The bigger the better.",
    yaxis_title="USD",
    showlegend=False,
    margin=dict(t=60, b=40),
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# 6) Optional: show the two reserve curves as a quick table preview
# ---------------------------------------------------------------------------
with st.expander("See sample of reserves per booking (preview)"):
    # Build the base table (uses variables defined earlier in the page)
    flat_series = pd.Series(float(flat_pct), index=df.index)  # flat % per row

    table_df = pd.DataFrame({
        "merchant_id": df.get("merchant_id", pd.Series("", index=df.index)).astype(str),
        "vertical": df.get("vertical", pd.Series("", index=df.index)).astype(str),
        "booking_amount": amount,
        "risk_probability": p,
        "flat_reserve_%": flat_series,
        "dynamic_reserve_%": pd.to_numeric(dyn_pct, errors="coerce").fillna(0.0),
    })

    # ---- Filters UI ----
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        merchant_q = st.text_input(
            "Filter by merchant_id (comma-separated; partial match OK)",
            placeholder="e.g., M0123, M045"
        )
    with f2:
        # Collect unique verticals (case-insensitive label)
        vertical_options = sorted(
            table_df["vertical"].dropna().astype(str).str.strip().unique().tolist()
        )
        selected_verticals = st.multiselect(
            "Filter by vertical",
            options=vertical_options,
            placeholder="Pick one or more"
        )
    with f3:
        max_rows = int(min(1000, len(table_df)))
        show_n = st.number_input("Rows", min_value=10, max_value=max_rows, value=min(50, max_rows), step=10)

    # ---- Apply filters ----
    mask = pd.Series(True, index=table_df.index)

    # Merchant filter (supports multiple tokens, partial, case-insensitive)
    if merchant_q.strip():
        toks = [t.strip().lower() for t in merchant_q.split(",") if t.strip()]
        mask &= table_df["merchant_id"].str.lower().apply(
            lambda x: any(tok in x for tok in toks)
        )

    # Vertical filter (exact match against selected values)
    if selected_verticals:
        mask &= table_df["vertical"].isin(selected_verticals)

    filtered = table_df[mask]

    display_df = filtered.head(int(show_n))

    # Small summary + table
    st.caption(f"Showing {len(display_df):,} of {len(filtered):,} rows")
    st.dataframe(filtered.head(int(show_n)), use_container_width=True)

    # Download filtered CSV
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="reserves_preview_filtered.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# 7) Footer explainer
# ---------------------------------------------------------------------------
with st.expander("📘 Glossary"):
    st.markdown("""
- **Reserves held ($)** — Cash you withhold from payouts under a policy.
- **% of volume withheld** — Reserves held divided by GMV; shows the liquidity impact on merchants.
- **Loss absorbed ($)** — Expected loss that is covered by reserves (cannot exceed the reserves you hold).
- **Net loss after reserves ($)** — What you still expect to lose after applying reserves.
- **Flat policy** — One reserve percentage applied to every merchant/booking.
- **Dynamic policy** — Model-recommended reserve percentage that varies by booking risk (more where risk is higher, less where it’s lower).
""")
