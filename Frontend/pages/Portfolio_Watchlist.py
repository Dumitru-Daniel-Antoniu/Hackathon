import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

st.title("📋 Portfolio Watchlist")
st.caption(
    "See **where potential non-delivery risk sits** across your portfolio and **which merchants drive it most** "
    "(See **Glossary** at the bottom of the page for detailed business definitions)."
)

df = st.session_state.get("scored_df")
if df is None:
    st.warning("No data yet. Upload a CSV on **Upload & Data Check** first.")
    st.stop()

# ---- Normalize & core fields
df = df.copy()
df.columns = [c.lower() for c in df.columns]
df["risk_probability"] = pd.to_numeric(df.get("risk_probability", df.get("risk_score", 0.0)), errors="coerce").fillna(0.0)

# Money at risk ("Exposure")
if "deposit_policy_percent" in df.columns:
    dep = pd.to_numeric(df["deposit_policy_percent"], errors="coerce").fillna(0.0) / 100.0
    df["exposure_$"] = (1.0 - dep).clip(0, 1) * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)
else:
    df["exposure_$"] = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)

# Estimated loss = chance × money at risk
df["estimated_loss_$"] = (df["risk_probability"] * df["exposure_$"]).fillna(0.0)

# ---- Filters (kept simple)
with st.expander("Filters"):
    i_sel = st.multiselect("Individual merchants", sorted(df["merchant_id"].astype(str).unique()))
    v_sel = st.multiselect("Business type (vertical)", sorted(df["vertical"].astype(str).unique()))
    c_sel = st.multiselect("Buyer country", sorted(df["country"].astype(str).unique()))
    dfv = df.copy()
    if i_sel: dfv = dfv[dfv["merchant_id"].astype(str).isin(i_sel)]
    if v_sel: dfv = dfv[dfv["vertical"].astype(str).isin(v_sel)]
    if c_sel: dfv = dfv[dfv["country"].astype(str).isin(c_sel)]

# ---- KPI row (plain-language + tooltips)
fmt_money = lambda x: f"${x:,.0f}"
k1, k2, k3, k4 = st.columns(4)
k1.metric("Merchants", dfv["merchant_id"].nunique(), help="Distinct merchants in the filtered view.")
k2.metric("Bookings", len(dfv), help="Number of transactions in the filtered view.")
k3.metric("Avg risk chance of non-delivery", f"{dfv['risk_probability'].mean():.2%}",
          help="Average model-predicted chance that a booking will not be fulfilled.")
k4.metric("Potential recovery (approx)", fmt_money(dfv["estimated_loss_$"].sum()),
          help="Sum of (chance × money at risk) across all filtered bookings.")

st.markdown("")

# ---- Chart 1: Risk concentration by business type
st.subheader("Where the **money at risk** can be **regained** (by business type)")
st.caption(
    "Bars show **estimated dollars to save** by business type (vertical). "
    "Taller bars = **More potential money** that can be recovered in that category."
)
by_vert = (dfv.groupby("vertical", as_index=False)
             .agg(Estimated_Loss_USD=("estimated_loss_$", "sum"),
                  Money_at_Risk_USD=("exposure_$", "sum"))
             .sort_values("Estimated_Loss_USD", ascending=False))

fig1 = px.bar(
    by_vert.head(12),
    x="vertical", y="Estimated_Loss_USD", text="Estimated_Loss_USD",
    labels={"vertical":"Business type", "Estimated_Loss_USD":"Potential gain ($)"},
    title=None,
)
fig1.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                   hovertemplate="<b>%{x}</b><br>Estimated gain ($) = %{y:,.2f}<extra></extra>",
                   cliponaxis=False)
st.plotly_chart(fig1, use_container_width=True)

# ---- Table: Top merchants driving estimated loss
st.subheader("Top merchants by **estimated potential gain**")
st.caption("This helps you **prioritize outreach / policy changes** where it matters most.")

# Friendly fallbacks if suggested_* columns aren’t present
avg_reserve_col = "suggested_reserve_percent" if "suggested_reserve_percent" in dfv.columns else None
avg_delay_col   = "suggested_settlement_delay_days" if "suggested_settlement_delay_days" in dfv.columns else None

agg = (dfv.groupby(["merchant_id","vertical"], as_index=False)
         .agg(
              bookings=("merchant_id","count"),
              gmv_usd=("booking_amount","sum"),
              avg_chance=("risk_probability","mean"),
              est_loss_usd=("estimated_loss_$","sum"),
              avg_reserve=(avg_reserve_col, "mean") if avg_reserve_col else ("risk_probability","mean"),
              avg_delay=(avg_delay_col, "mean") if avg_delay_col else ("risk_probability","mean"),
          )
         .sort_values("est_loss_usd", ascending=False))

# Display-only renames + formatting
display = agg.rename(columns={
    "merchant_id":"Merchant",
    "vertical":"Business type",
    "bookings":"Bookings",
    "gmv_usd":"GMV - Gross Merchandise Value ($)",
    "avg_chance":"Avg suggested risk chance",
    "est_loss_usd":"Estimated potential gain ($)",
    "avg_reserve":"Avg suggested funds held (%)",
    "avg_delay":"Avg suggested payout delay (days)",
})
display["GMV - Gross Merchandise Value ($)"] = display["GMV - Gross Merchandise Value ($)"].round(0).astype(int).map(lambda x: f"{x:,}")
display["Estimated potential gain ($)"] = display["Estimated potential gain ($)"].round(0).astype(int).map(lambda x: f"{x:,}")
display["Avg suggested risk chance"] = (display["Avg suggested risk chance"]*100).round(1).map(lambda x: f"{x:.1f}%")
if "Avg funds held (%)" in display.columns:
    display["Avg suggested funds held (%)"] = display["Avg suggested funds held (%)"].astype(float).round(2)
if "Avg payout delay (days)" in display.columns:
    display["Avg suggested payout delay (days)"] = display["Avg suggested payout delay (days)"].astype(float).round(1)

st.dataframe(display.head(20), use_container_width=True)

# ---- Heatmap: 10-day horizon buckets × risk tier
st.subheader("When the risk shows up (service horizon × risk tier)")
st.caption(
    "Which combinations of **booking timing** and **merchant risk level** have the most **money to regain**"
)

# 1) Prepare bins and labels
bins = list(range(0, 200, 10))                     # 0,10,...,190
labels = [f"{b}-{b+10}" for b in bins]             # "0-10","10-20",...,"190-200"

d = dfv.copy()
d["horizon_raw"] = pd.to_numeric(d["days_in_advance"], errors="coerce").fillna(0)
d = d[d["horizon_raw"] <= 200].copy()              # clamp view to 0..200

# put day 200 into the 190–200 bin by clipping to 199 before flooring
bucket = (d["horizon_raw"].clip(upper=199) // 10 * 10).astype(int)
d["horizon_label"] = bucket.astype(str) + "-" + (bucket + 10).astype(str)

# enforce risk-tier order if available
tiers_order = [
    "Trusted Partner",
    "Established Operator",
    "Developing Organization",
    "High-Risk Counterparty",
    "Fraudulent Actor",
]
if "risk_tier" in d.columns:
    d["risk_tier"] = pd.Categorical(d["risk_tier"], categories=tiers_order, ordered=True)
else:
    d["risk_tier"] = "All"

# 2) Aggregate to a matrix (risk_tier × 10-day label)
matrix = (
    d.groupby(["risk_tier", "horizon_label"], observed=True)["exposure_$"]
     .sum()
     .unstack(fill_value=0)
)

# guarantee full grid ordering (missing combos become 0)
matrix = matrix.reindex(index=tiers_order if "risk_tier" in d.columns else ["All"], fill_value=0)
matrix = matrix.reindex(columns=labels, fill_value=0)

# Hide tiles where money at risk == 0
matrix = matrix.astype(float).where(matrix != 0, np.nan)

matrix = matrix.dropna(how="all", axis=0)  # drop empty risk tiers
matrix = matrix.dropna(how="all", axis=1)  # drop empty horizon buckets

# 3) Plot as a true block heatmap (discrete columns)
fig2 = px.imshow(
    matrix,
    aspect="auto",
    color_continuous_scale="Reds",
    labels=dict(x="Days in advance", y="Risk tier", color="Money to regain ($)"),
)

# Cleaner hover + colorbar formatting
fig2.update_traces(hoverongaps=False, colorbar=dict(tickformat="$,.0f"))

st.plotly_chart(fig2, use_container_width=True)

# ---- Download
st.download_button(
    "⬇️ Download watchlist (CSV)",
    data=agg.to_csv(index=False).encode("utf-8"),
    file_name="merchant_watchlist.csv",
    mime="text/csv",
)

# ---- Glossary
with st.expander("Glossary"):
    st.markdown(
        """
- **GMV ($)** – Total value of bookings in dollars (before reserves or refunds).
- **Money at risk (Exposure)** – Dollars that could be lost **if** a booking is not delivered.
  Calculated as `(1 − deposit%) × booking amount` (or just the amount if deposit% is missing).
- **Estimated loss ($)** – **Chance of non-delivery × money at risk**. Summed to show where losses concentrate.
- **Avg funds held (%)** – Model-suggested **reserve percentage** held to protect against non-delivery.
- **Avg payout delay (days)** – Model-suggested **days to delay settlement** (payout) to reduce risk.
- **Risk tier** – Business-friendly label derived from the model’s probability (Trusted → Fraudulent).
        """
    )
