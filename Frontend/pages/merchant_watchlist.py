# Frontend/pages/merchant_watchlist.py
import streamlit as st
import pandas as pd
import numpy as np

st.header("🚨 Merchant Watchlist")

uploaded = st.file_uploader(
    "Upload the *enriched* CSV from the Upload & Score tab (scored_bookings.csv)",
    type=["csv"],
)

if not uploaded:
    st.info("Upload your scored CSV to view merchant risk.")
    st.stop()

df = pd.read_csv(uploaded)

# ---------- Normalize column names (handles both lower & UPPER from backend) ----------
rename_if_upper = {
    "MERCHANT_ID": "merchant_id",
    "VERTICAL": "vertical",
    "COUNTRY": "country",
    "BOOKING_AMOUNT": "booking_amount",
    "DAYS_IN_ADVANCE": "days_in_advance",
    "NEW_MERCHANT": "new_merchant",
    "SHOCK_FLAG": "shock_flag",
}
present = {k: k for k in df.columns}
need_rename = {k: v for k, v in rename_if_upper.items() if k in present and v not in present}
if need_rename:
    df = df.rename(columns=need_rename)

# Required minimal columns
required = ["merchant_id", "vertical", "booking_amount", "risk_score", "risk_tier"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

# Expected loss convenience column (if not already there)
if "expected_loss$" not in df.columns:
    df["expected_loss$"] = df["risk_score"].astype(float) * df["booking_amount"].astype(float)

# High-risk flag aligned with your backend tiers / thresholds
# Backend tiers: Trusted Partner, Established Operator, Developing Organization,
#                High-Risk Counterparty, Fraudulent Actor
# Treat top two tiers as "high".
high_tiers = {"High-Risk Counterparty", "Fraudulent Actor"}
df["high_flag"] = (df["risk_score"].astype(float) >= 0.41) | (df["risk_tier"].isin(high_tiers))

# ---------- Portfolio KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Merchants", df["merchant_id"].nunique())
col2.metric("Bookings", len(df))
col3.metric("Avg risk", f"{df['risk_score'].mean():.2%}")
col4.metric("Expected loss ($)", f"{(df['expected_loss$']).sum():,.0f}")

# ---------- Filters ----------
with st.expander("Filters"):
    merch_sel = st.multiselect("Merchant(s)", sorted(df["merchant_id"].astype(str).unique()))
    vert_sel  = st.multiselect("Vertical(s)", sorted(df["vertical"].astype(str).unique()))
    dfv = df.copy()
    if merch_sel:
        dfv = dfv[dfv["merchant_id"].astype(str).isin(merch_sel)]
    if vert_sel:
        dfv = dfv[dfv["vertical"].astype(str).isin(vert_sel)]

# ---------- Aggregation ----------
agg = (
    dfv.groupby(["merchant_id", "vertical"], dropna=False)
      .agg(
        bookings=("merchant_id", "count"),
        amount_usd=("booking_amount", "sum"),
        avg_risk=("risk_score", "mean"),
        high_share=("high_flag", "mean"),
        expected_loss_usd=("expected_loss$", "sum"),
        avg_sugg_reserve=("suggested_reserve_percent", "mean") if "suggested_reserve_percent" in dfv.columns else ("risk_score", "mean"),
        avg_sugg_delay=("suggested_settlement_delay_days", "mean") if "suggested_settlement_delay_days" in dfv.columns else ("risk_score", "mean"),
      )
      .sort_values("expected_loss_usd", ascending=False)
      .reset_index()
)

st.subheader("Top merchants by expected loss")
st.dataframe(
    agg.head(20).assign(
        avg_risk=lambda x: (x["avg_risk"] * 100).round(1).astype(str) + "%",
        high_share=lambda x: (x["high_share"] * 100).round(1).astype(str) + "%",
        avg_sugg_reserve=lambda x: x["avg_sugg_reserve"].round(2),
        avg_sugg_delay=lambda x: x["avg_sugg_delay"].round(1),
        amount_usd=lambda x: x["amount_usd"].round(0).astype(int),
        expected_loss_usd=lambda x: x["expected_loss_usd"].round(0).astype(int),
    ),
    use_container_width=True,
)

# ---------- Simple charts ----------
st.subheader("Exposure (expected loss) — Top 10 merchants")
top10 = agg.head(10)[["merchant_id", "expected_loss_usd"]].copy()
top10 = top10.rename(columns={"merchant_id": "Merchant", "expected_loss_usd": "Expected loss ($)"})
st.bar_chart(top10.set_index("Merchant"))

st.subheader("Average risk by vertical")
by_vert = agg.groupby("vertical", dropna=False)["avg_risk"].mean().sort_values(ascending=False)
st.bar_chart(by_vert)

# ---------- Drill-down ----------
st.subheader("Merchant drill-down")
pick = st.selectbox("Choose a merchant", ["—"] + list(agg["merchant_id"].astype(str).unique()))
if pick and pick != "—":
    sub = dfv[dfv["merchant_id"].astype(str) == pick].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Bookings", len(sub))
    c2.metric("Avg risk", f"{sub['risk_score'].mean():.2%}")
    c3.metric("Expected loss ($)", f"{(sub['expected_loss$']).sum():,.0f}")

    st.write("Top risky bookings")
    cols = [
        "vertical", "booking_amount", "days_in_advance", "risk_score",
        "risk_tier", "suggested_reserve_percent", "suggested_settlement_delay_days"
    ]
    cols = [c for c in cols if c in sub.columns]
    st.dataframe(
        sub.sort_values("risk_score", ascending=False)[cols].head(200),
        use_container_width=True
    )

# ---------- Export ----------
st.download_button(
    "⬇️ Download merchant watchlist (CSV)",
    data=agg.to_csv(index=False).encode("utf-8"),
    file_name="merchant_watchlist.csv",
    mime="text/csv",
)
