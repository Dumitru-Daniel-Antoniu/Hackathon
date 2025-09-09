# Frontend/pages/Executive_Summary.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("📊 Executive Summary")

df = st.session_state.get("scored_df")
if df is None:
    st.warning("No data yet. Upload a CSV on **Upload & Data Check** first.")
    st.stop()

df = df.copy()
df.columns = [c.lower() for c in df.columns]

# ---- Business assumptions (kept simple, adjustable) ----
with st.expander("Assumptions (used to translate policy into loss reduction)"):
    alpha = st.slider("Effect of Funds Held (α)", 0.0, 1.0, 0.60, 0.01,
                      help="How strongly reserve% reduces expected loss.")
    beta = st.slider("Effect of Payout Delay per day (β)", 0.0, 0.05, 0.015, 0.001,
                     help="How strongly each day of payout delay reduces expected loss.")
    baseline_reserve = st.slider("Baseline flat reserve %", 0, 50, 10, 1)
    baseline_delay   = st.slider("Baseline payout delay (days)", 0, 45, 0, 1)
    capital_rate     = st.slider("Cost of capital (annual %)", 0.00, 0.20, 0.08, 0.01)

# Exposure proxy: (1 - deposit%) * amount  (fallback to amount)
if "deposit_policy_percent" in df.columns:
    dep = pd.to_numeric(df["deposit_policy_percent"], errors="coerce").fillna(0.0)/100.0
    exposure = (1.0 - dep).clip(0,1) * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)
else:
    exposure = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)

prob = pd.to_numeric(df.get("probability", df.get("risk_score", 0.0)), errors="coerce").fillna(0.0)
rsv_model = pd.to_numeric(df.get("suggested_reserve_percent", 0.0), errors="coerce").fillna(0.0)
dly_model = pd.to_numeric(df.get("suggested_settlement_delay_days", 0), errors="coerce").fillna(0)

# Translate policy into reduced risk (simple linear effect; demo-ready)
reduction_model = (alpha * (rsv_model/100.0) + beta * dly_model).clip(0, 0.95)
risk_after_model = prob * (1 - reduction_model)

# Baseline
rsv_base = baseline_reserve/100.0
dly_base = baseline_delay
reduction_base = (alpha * rsv_base + beta * dly_base)
risk_after_base = prob * (1 - min(reduction_base, 0.95))

# KPIs
gmv = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0).sum()
el_model = (risk_after_model * exposure).sum()
el_base  = (risk_after_base  * exposure).sum()
held_model = (rsv_model/100.0 * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)).sum()
held_base  = (baseline_reserve/100.0 * pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)).sum()
reserve_cost_model = held_model * (capital_rate * (pd.to_numeric(dly_model, errors="coerce").fillna(0).mean()/365.0))
reserve_cost_base  = held_base  * (capital_rate * (baseline_delay/365.0))

fmt = lambda x: f"${x:,.0f}"
c1,c2,c3,c4 = st.columns(4)
c1.metric("GMV analyzed", fmt(gmv))
c2.metric("Expected Loss (model)", fmt(el_model), delta=f"{fmt(el_base - el_model)} vs baseline")
c3.metric("Funds Held (model)", fmt(held_model), delta=f"{fmt(held_model - held_base)} vs baseline")
c4.metric("Avg payout delay (days)", f"{dly_model.mean():.1f}", delta=f"{dly_model.mean()-baseline_delay:+.1f} vs baseline")

# ROI bars
roi = go.Figure()
roi.add_bar(name="Expected Loss – Baseline", x=["Baseline"], y=[el_base])
roi.add_bar(name="Expected Loss – Model",    x=["Model"],    y=[el_model])
roi.add_bar(name="Funds Held – Baseline",    x=["Baseline"], y=[held_base])
roi.add_bar(name="Funds Held – Model",       x=["Model"],    y=[held_model])
roi.update_layout(barmode="group", title="ROI: Expected Loss vs Funds Held (lower is better)", yaxis_title="$")
st.plotly_chart(roi, use_container_width=True)

# Top actions (merchants that drive most benefit)
df_work = df.copy()
df_work["exposure"] = exposure
df_work["risk_after_model"] = risk_after_model
df_work["risk_after_base"]  = risk_after_base
df_work["loss_model"] = df_work["risk_after_model"] * df_work["exposure"]
df_work["loss_base"]  = df_work["risk_after_base"]  * df_work["exposure"]
gain = (df_work.groupby(["merchant_id","vertical"])
              .agg(
                  bookings=("merchant_id","count"),
                  gmv=("booking_amount","sum"),
                  loss_model=("loss_model","sum"),
                  loss_base=("loss_base","sum"),
                  avg_reserve=("suggested_reserve_percent","mean"),
                  avg_delay=("suggested_settlement_delay_days","mean")
              )
              .assign(loss_avoided=lambda x: x["loss_base"] - x["loss_model"])
              .sort_values("loss_avoided", ascending=False).reset_index())

st.subheader("Top actions — where we reduce loss the most")
st.dataframe(
    gain[["merchant_id","vertical","bookings","gmv","loss_avoided","avg_reserve","avg_delay"]].head(10)
        .assign(gmv=lambda x: x["gmv"].round(0).astype(int),
                loss_avoided=lambda x: x["loss_avoided"].round(0).astype(int),
                avg_reserve=lambda x: x["avg_reserve"].round(2),
                avg_delay=lambda x: x["avg_delay"].round(1)),
    use_container_width=True
)

st.caption("Assumptions (α, β) align with the prototype used on the Merchant Explorer. Adjust them above to stress-test.")
