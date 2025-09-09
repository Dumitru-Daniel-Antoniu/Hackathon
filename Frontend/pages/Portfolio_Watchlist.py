# Frontend/pages/Portfolio_Watchlist.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📋 Portfolio Watchlist")

df = st.session_state.get("scored_df")
if df is None:
    st.warning("No data yet. Upload a CSV on **Upload & Data Check** first.")
    st.stop()

df = df.copy()
df.columns = [c.lower() for c in df.columns]

# Exposure proxy
if "deposit_policy_percent" in df.columns:
    dep = pd.to_numeric(df["deposit_policy_percent"], errors="coerce").fillna(0.0)/100.0
    df["exposure"] = (1.0-dep).clip(0,1) * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)
else:
    df["exposure"] = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)

# Risk prob
df["probability"] = pd.to_numeric(df.get("probability", df.get("risk_score", 0.0)), errors="coerce").fillna(0.0)
df["expected_loss$"] = (df["probability"] * df["exposure"]).fillna(0.0)

# Filters
with st.expander("Filters"):
    v_sel = st.multiselect("Vertical(s)", sorted(df["vertical"].astype(str).unique()))
    c_sel = st.multiselect("Country(ies)", sorted(df["country"].astype(str).unique()))
    dfv = df.copy()
    if v_sel: dfv = dfv[dfv["vertical"].astype(str).isin(v_sel)]
    if c_sel: dfv = dfv[dfv["country"].astype(str).isin(c_sel)]

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Merchants", dfv["merchant_id"].nunique())
c2.metric("Bookings", len(dfv))
c3.metric("Avg chance of non-delivery", f"{dfv['probability'].mean():.2%}")
c4.metric("Expected loss (approx)", f"${dfv['expected_loss$'].sum():,.0f}")

# Concentration by vertical (bar)
by_vert = (dfv.groupby("vertical", as_index=False)
             .agg(gmv=("booking_amount","sum"), expected_loss=("expected_loss$","sum"))
             .sort_values("expected_loss", ascending=False))
fig1 = px.bar(by_vert.head(12), x="vertical", y="expected_loss", text="expected_loss",
              labels={"expected_loss":"Expected loss ($)", "vertical":""},
              title="Where exposure concentrates (by vertical)")
fig1.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False)
st.plotly_chart(fig1, use_container_width=True)

# Top merchants
agg = (dfv.groupby(["merchant_id","vertical"], as_index=False)
         .agg(bookings=("merchant_id","count"),
              gmv=("booking_amount","sum"),
              avg_risk=("probability","mean"),
              expected_loss=("expected_loss$","sum"),
              avg_reserve=("suggested_reserve_percent","mean") if "suggested_reserve_percent" in dfv.columns else ("probability","mean"),
              avg_delay=("suggested_settlement_delay_days","mean") if "suggested_settlement_delay_days" in dfv.columns else ("probability","mean"))
         .sort_values("expected_loss", ascending=False))

st.subheader("Top merchants by expected loss")
st.dataframe(
    agg.head(20).assign(
        gmv=lambda x: x["gmv"].round(0).astype(int),
        expected_loss=lambda x: x["expected_loss"].round(0).astype(int),
        avg_risk=lambda x: (x["avg_risk"]*100).round(1).astype(str) + "%"
    ),
    use_container_width=True
)

# Horizon heatmap (service horizon bucket × risk tier if available)
if "risk_tier" in dfv.columns:
    tiers_order = [
        "Trusted Partner",
        "Established Operator",
        "Developing Organization",
        "High-Risk Counterparty",
        "Fraudulent Actor",
    ]
    dfv["horizon_bucket"] = (pd.to_numeric(dfv["days_in_advance"], errors="coerce").fillna(0)//30)*30
    dfv["risk_tier"] = pd.Categorical(dfv["risk_tier"],
                                      categories=tiers_order,
                                      ordered=True)
    heat = (dfv.groupby(["horizon_bucket", "risk_tier"], observed=True, as_index=False)
            .agg(exposure=("exposure", "sum")))
    fig2 = px.density_heatmap(heat, x="horizon_bucket", y="risk_tier", z="exposure",
                              category_orders={"risk_tier": tiers_order},
                              nbinsx=12, title="Exposure by horizon & risk tier",
                              labels={"horizon_bucket":"Days in advance","risk_tier":"Risk tier","exposure":"Exposure ($)"})
    st.plotly_chart(fig2, use_container_width=True)

# Export watchlist
st.download_button(
    "⬇️ Download watchlist (CSV)",
    data=agg.to_csv(index=False).encode("utf-8"),
    file_name="merchant_watchlist.csv",
    mime="text/csv",
)
