import streamlit as st, pandas as pd
from utilities.api import score_via_api

st.header("📄 Upload & Score (API mode)")
api_base = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

uploaded = st.file_uploader("Upload bookings CSV", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)

    try:
        scored = score_via_api(raw, api_base)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

    st.metric("Rows", len(scored))
    st.metric("Avg risk", f"{scored['risk_score'].mean():.2%}")
    st.dataframe(scored.head(300), use_container_width=True)
    st.download_button(
        "⬇️ Download enriched CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_bookings.csv",
        mime="text/csv"
    )
else:
    st.info("Upload a CSV to score.")
