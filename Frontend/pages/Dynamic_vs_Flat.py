# Frontend/pages/2_💡_Dynamic_vs_Flat.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Dynamic vs Flat Reserves", layout="wide")
st.title("💡 Dynamic vs Flat Reserves — How much more do we protect?")

st.caption(
    "This view compares a **single flat reserve%** applied to every merchant with our **model’s dynamic reserves**. "
    "We show **dollars protected** (loss avoided) and the **reserves held** to get there. "
    "Formulas: per booking, *Expected Loss* = `probability × booking_amount`; "
    "*Loss absorbed* = `min(Expected Loss, reserves)`; *Net loss left* = `Expected Loss − Loss absorbed` (never negative)."
)

# -----------------------------
# Load scored data from session
# -----------------------------
df = st.session_state.get("scored_df")
if df is None or df.empty:
    st.warning("No data in session. Go to **Upload & Data Check** first.")
    st.stop()

df = df.copy()
df.columns = [c.lower() for c in df.columns]

# Probability & amount
if "probability" not in df.columns:
    # Back-compat: derive from risk_score if needed
    rs = pd.to_numeric(df.get("risk_score_pct", 0.0), errors="coerce").fillna(0.0)
    df["probability"] = rs.where(rs <= 1.0, rs / 100.0).clip(0, 1)

amount = pd.to_numeric(df["booking_amount"], errors="coerce").fillna(0.0)
p = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0).clip(0, 1)
df["expected_loss"] = p * amount  # EL per booking

# Dynamic reserves from the model
if "suggested_reserve_percent" in df.columns:
    dyn_pct = pd.to_numeric(df["suggested_reserve_percent"], errors="coerce").fillna(0.0).clip(0, 100)
else:
    dyn_pct = pd.Series(0.0, index=df.index)
    st.warning("Column `suggested_reserve_percent` missing — dynamic policy assumed to be 0% (check your scoring).")

df["reserve_amt_dynamic"] = (dyn_pct / 100.0) * amount

# -----------------------------
# Sidebar — flat policy control
# -----------------------------
st.sidebar.header("Policy you want to compare")
flat_pct = st.sidebar.slider("Flat reserve % (same for every merchant)", 0, 50, 10, 1)
df["reserve_amt_flat"] = (flat_pct / 100.0) * amount

# -----------------------------
# Helpers to aggregate a policy
# -----------------------------
def summarize_policy(reserve_amt: pd.Series) -> dict:
    el = df["expected_loss"]
    loss_absorbed = np.minimum(el.values, reserve_amt.values)           # per booking
    net_left = (el.values - loss_absorbed)                              # per booking (>= 0)
    return {
        "reserves_held": float(reserve_amt.sum()),
        "loss_absorbed": float(loss_absorbed.sum()),
        "net_left": float(net_left.sum()),
    }

dyn = summarize_policy(df["reserve_amt_dynamic"])
flat = summarize_policy(df["reserve_amt_flat"])

portfolio_gmv = float(amount.sum())
total_el = float(df["expected_loss"].sum())

# Incremental story: how much more do we protect by going dynamic?
incremental_protection = dyn["loss_absorbed"] - flat["loss_absorbed"]
incremental_reserves   = dyn["reserves_held"] - flat["reserves_held"]
roi = (incremental_protection / incremental_reserves) if incremental_reserves > 0 else np.nan

# -----------------------------
# KPI strip — plain language
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portfolio volume (GMV)", f"${portfolio_gmv:,.0f}")
k2.metric("Total expected loss (EL)", f"${total_el:,.0f}")
k3.metric("Flat policy — Reserves held", f"${flat['reserves_held']:,.0f}", help="Single % across all merchants.")
k4.metric("Dynamic policy — Reserves held", f"${dyn['reserves_held']:,.0f}", help="Model-driven % per merchant.")

k5, k6, k7 = st.columns(3)
k5.metric("Dollars protected (flat)", f"${flat['loss_absorbed']:,.0f}")
k6.metric("Dollars protected (dynamic)", f"${dyn['loss_absorbed']:,.0f}")
k7.metric("Extra dollars protected by dynamic", f"${incremental_protection:,.0f}",
          delta=f"{(100*incremental_protection/max(1,total_el)):.1f}% of EL")

st.caption(
    f"**What this means:** With a flat {flat_pct}% reserve, you protect **${flat['loss_absorbed']:,.0f}**. "
    f"Using dynamic reserves, you protect **${dyn['loss_absorbed']:,.0f}** — that’s **${incremental_protection:,.0f} more**. "
    f"The dynamic policy leaves **${dyn['net_left']:,.0f}** of expected loss uncovered versus **${flat['net_left']:,.0f}** with the flat policy."
)

# -----------------------------
# Bars — protection vs reserves
# -----------------------------
bar = go.Figure()
bar.add_bar(name=f"Loss avoided (protected $)", x=["Flat", "Dynamic"],
            y=[flat["loss_absorbed"], dyn["loss_absorbed"]])
bar.add_bar(name=f"Reserves held ($)", x=["Flat", "Dynamic"],
            y=[flat["reserves_held"], dyn["reserves_held"]])
bar.update_layout(
    barmode="group",
    title=f"Protection vs Cost — Flat {flat_pct}% vs Dynamic",
    yaxis_title="$",
    legend_title="Measure",
)
st.plotly_chart(bar, use_container_width=True)

# -----------------------------
# ROI card — incremental view
# -----------------------------
with st.expander("How efficient is the extra buffer?", expanded=False):
    st.markdown(
        f"- **Extra dollars protected by dynamic (vs flat {flat_pct}%):** "
        f"**${incremental_protection:,.0f}**  \n"
        f"- **Extra reserves held (vs flat):** ${incremental_reserves:,.0f}  \n"
        f"- **ROI of extra reserves:** {('' if np.isfinite(roi) else 'n/a ')}"
        f"{'' if not np.isfinite(roi) else f'${roi:,.2f} protected per $1 of extra reserves'}"
    )
    st.caption(
        "Interpretation: ROI > 1 means every additional $1 reserved by the dynamic policy protects more than $1 of expected loss in aggregate."
    )

# -----------------------------
# Presentation-ready table
# -----------------------------
def percent(x):
    return f"{(x/portfolio_gmv*100):.1f}%" if portfolio_gmv > 0 else "0.0%"

summary = pd.DataFrame([
    {"Policy": f"Flat {flat_pct}%", "Reserves held ($)": flat["reserves_held"],
     "% of volume held": percent(flat["reserves_held"]),
     "Loss avoided ($)": flat["loss_absorbed"],
     "% of EL absorbed": f"{(flat['loss_absorbed']/total_el*100 if total_el>0 else 0):.1f}%",
     "Net loss left ($)": flat["net_left"]},
    {"Policy": "Dynamic (model)", "Reserves held ($)": dyn["reserves_held"],
     "% of volume held": percent(dyn["reserves_held"]),
     "Loss avoided ($)": dyn["loss_absorbed"],
     "% of EL absorbed": f"{(dyn['loss_absorbed']/total_el*100 if total_el>0 else 0):.1f}%",
     "Net loss left ($)": dyn["net_left"]},
])

# Friendly formatting for display
disp = summary.copy()
for col in ["Reserves held ($)","Loss avoided ($)","Net loss left ($)"]:
    disp[col] = disp[col].map(lambda v: f"${v:,.0f}")
st.subheader("Policy comparison (portfolio totals)")
st.dataframe(disp, use_container_width=True)

# -----------------------------
# One-sentence executive takeaway
# -----------------------------
delta_saved = flat["net_left"] - dyn["net_left"]
st.success(
    f"**Executive takeaway:** Switching from a flat {flat_pct}% to the dynamic policy would protect "
    f"**${incremental_protection:,.0f} more** of expected loss, reducing uncovered risk by **${delta_saved:,.0f}**."
)

# Optional download
st.download_button(
    "⬇️ Download comparison (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=f"dynamic_vs_flat_{flat_pct}pct.csv",
    mime="text/csv",
)
