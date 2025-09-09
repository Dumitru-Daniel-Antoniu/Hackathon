# Frontend/app.py
import streamlit as st

st.set_page_config(page_title="PrePayGuard", layout="wide")

# keep a shared backend URL + scored df in session
if "api_base" not in st.session_state:
    st.session_state["api_base"] = "http://localhost:8000"
if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = None

st.title("PrePayGuard")
st.caption("Upload once → see business impact. Use the left sidebar to open pages.")
