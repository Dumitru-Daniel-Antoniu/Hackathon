# Frontend/pages/3_📋_Portfolio_Watchlist.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📋 Portfolio Watchlist")
st.caption(
    "See **where potential non-delivery risk sits** across your portfolio and **which merchants drive it most**. "
    "We use simple business definitions throughout (see *Glossary* below)."
)

df = st.session_state.get("scored_df")
if df is None:
    st.warning("No data yet. Upload a CSV on **Upload & Data Check** first.")
    st.stop()

# ---- Normalize & core fields
df = df.copy()
df.columns = [c.lower() for c in df.columns]
df["probability"] = pd.to_numeric(df.get("probability", df.get("risk_score", 0.0)), errors="coerce").fillna(0.0)

# Money at risk ("Exposure")
if "deposit_policy_percent" in df.columns:
    dep = pd.to_numeric(df["deposit_policy_percent"], errors="coerce").fillna(0.0) / 100.0
    df["exposure_$"] = (1.0 - dep).clip(0, 1) * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)
else:
    df["exposure_$"] = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)

# Estimated loss = chance × money at risk
df["estimated_loss_$"] = (df["probability"] * df["exposure_$"]).fillna(0.0)

# ---- Filters (kept simple)
with st.expander("Filters"):
    v_sel = st.multiselect("Business type (vertical)", sorted(df["vertical"].astype(str).unique()))
    c_sel = st.multiselect("Buyer country", sorted(df["country"].astype(str).unique()))
    dfv = df.copy()
    if v_sel: dfv = dfv[dfv["vertical"].astype(str).isin(v_sel)]
    if c_sel: dfv = dfv[dfv["country"].astype(str).isin(c_sel)]

# ---- KPI row (plain-language + tooltips)
fmt_money = lambda x: f"${x:,.0f}"
k1, k2, k3, k4 = st.columns(4)
k1.metric("Merchants", dfv["merchant_id"].nunique(), help="Distinct merchants in the filtered view.")
k2.metric("Bookings", len(dfv), help="Number of transactions in the filtered view.")
k3.metric("Avg chance of non-delivery", f"{dfv['probability'].mean():.2%}",
          help="Average model-predicted chance that a booking will not be fulfilled.")
k4.metric("Estimated loss (approx)", fmt_money(dfv["estimated_loss_$"].sum()),
          help="Sum of (chance × money at risk) across all filtered bookings.")

st.markdown("")

# ---- Chart 1: Risk concentration by business type
st.subheader("Where the **money at risk** concentrates (by business type)")
st.caption(
    "Bars show **estimated loss in dollars** by business type (vertical). "
    "Taller bars = **more dollars at risk** in that category."
)
by_vert = (dfv.groupby("vertical", as_index=False)
             .agg(Estimated_Loss_USD=("estimated_loss_$", "sum"),
                  Money_at_Risk_USD=("exposure_$", "sum"))
             .sort_values("Estimated_Loss_USD", ascending=False))

fig1 = px.bar(
    by_vert.head(12),
    x="vertical", y="Estimated_Loss_USD", text="Estimated_Loss_USD",
    labels={"vertical":"Business type", "Estimated_Loss_USD":"Estimated loss ($)"},
    title=None,
)
fig1.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False)
st.plotly_chart(fig1, use_container_width=True)

# ---- Table: Top merchants driving estimated loss
st.subheader("Top merchants by **estimated loss**")
st.caption("This helps you **prioritize outreach / policy changes** where it matters most.")

# Friendly fallbacks if suggested_* columns aren’t present
avg_reserve_col = "suggested_reserve_percent" if "suggested_reserve_percent" in dfv.columns else None
avg_delay_col   = "suggested_settlement_delay_days" if "suggested_settlement_delay_days" in dfv.columns else None

agg = (dfv.groupby(["merchant_id","vertical"], as_index=False)
         .agg(
              bookings=("merchant_id","count"),
              gmv_usd=("booking_amount","sum"),
              avg_chance=("probability","mean"),
              est_loss_usd=("estimated_loss_$","sum"),
              avg_reserve=(avg_reserve_col, "mean") if avg_reserve_col else ("probability","mean"),
              avg_delay=(avg_delay_col, "mean") if avg_delay_col else ("probability","mean"),
          )
         .sort_values("est_loss_usd", ascending=False))

# Display-only renames + formatting
display = agg.rename(columns={
    "merchant_id":"Merchant",
    "vertical":"Business type",
    "bookings":"Bookings",
    "gmv_usd":"GMV ($)",
    "avg_chance":"Avg chance",
    "est_loss_usd":"Estimated loss ($)",
    "avg_reserve":"Avg funds held (%)",
    "avg_delay":"Avg payout delay (days)",
})
display["GMV ($)"] = display["GMV ($)"].round(0).astype(int).map(lambda x: f"{x:,}")
display["Estimated loss ($)"] = display["Estimated loss ($)"].round(0).astype(int).map(lambda x: f"{x:,}")
display["Avg chance"] = (display["Avg chance"]*100).round(1).map(lambda x: f"{x:.1f}%")
if "Avg funds held (%)" in display.columns:
    display["Avg funds held (%)"] = display["Avg funds held (%)"].astype(float).round(2)
if "Avg payout delay (days)" in display.columns:
    display["Avg payout delay (days)"] = display["Avg payout delay (days)"].astype(float).round(1)

st.dataframe(display.head(20), use_container_width=True)

# ---- Heatmap: 10-day horizon buckets × risk tier
st.subheader("When the risk shows up (service horizon × risk tier)")
st.caption(
    "Each column is a **10-day bin** (0–10, 10–20, …, 190–200). "
    "Cells show **money at risk** for that bin and risk tier."
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

# 3) Plot as a true block heatmap (discrete columns)
fig2 = px.imshow(
    matrix,
    aspect="auto",
    color_continuous_scale="Reds",
    labels=dict(x="Days in advance (10-day bins)", y="Risk tier", color="Money at risk ($)"),
)
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
