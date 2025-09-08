import streamlit as st, pandas as pd

st.header("🚨 Merchant Watchlist")

uploaded = st.file_uploader("Upload the *enriched* CSV from the Upload & Score tab", type=["csv"], key="watch")
if not uploaded:
    st.info("Upload scored_bookings.csv to view merchant risk.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Portfolio overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Merchants", df["merchant_id"].nunique())
c2.metric("Avg risk", f"{df['risk_score'].mean():.2%}")
c3.metric("High-risk merchants", f"{(df.groupby('merchant_id')['risk_tier'].apply(lambda s: (s=='high').mean()>0.2)).sum()}")
c4.metric("Expected loss ($)", f"{(df['risk_score']*df['booking_amount']).sum():,.0f}")

st.subheader("Top merchants by expected loss")
agg = (df.groupby(["merchant_id","vertical"])
         .agg(
            bookings=("merchant_id","count"),
            amount_usd=("booking_amount","sum"),
            avg_risk=("risk_score","mean"),
            high_share=("risk_tier", lambda s: (s=="high").mean()),
            expected_loss_usd=("expected_loss$", "sum"),
         )
         .sort_values("expected_loss_usd", ascending=False)
         .reset_index())
st.dataframe(agg.head(20), use_container_width=True)

# Simple chart (expected loss by top 10 merchants)
top10 = agg.head(10).set_index("merchant_id")["expected_loss_usd"]
st.bar_chart(top10)