
import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="Forward‑Delivery Risk Studio",
    page_icon="💳",
    layout="wide",
)

ASSETS = Path(__file__).parent / "assets"

def _set_background():
    # Embed SVG as CSS background
    svg_path = ASSETS / "background.svg"
    if svg_path.exists():
        svg_bytes = svg_path.read_bytes()
        b64 = base64.b64encode(svg_bytes).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/svg+xml;base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        /* Glassy cards */
        .hero {{
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 18px;
            padding: 2.2rem 2.4rem;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}
        .pill {{
            display:inline-block;
            padding: .25rem .6rem;
            font-size:.78rem;
            border-radius:999px;
            background:rgba(255,255,255,0.18);
            border:1px solid rgba(255,255,255,0.25);
            margin-right:.6rem;
        }}
        .small-muted {{
            color: rgba(255,255,255,0.85);
            font-size: 0.95rem;
        }}
        .section-card {{
            background: rgba(255,255,255,0.82);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(0,0,0,0.05);
        }}
        .section-title {{
            margin: 0;
            padding: 0;
        }}
        hr.soft {{
            border: none;
            height: 1px;
            background: linear-gradient(to right, rgba(0,0,0,0.0), rgba(0,0,0,0.35), rgba(0,0,0,0.0));
            margin: 1rem 0 1.2rem 0;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

_set_background()

st.markdown(
    """
    <div class="hero">
      <div class="pill">payments</div>
      <div class="pill">risk</div>
      <div class="pill">forward‑delivery</div>
      <h1 style="margin-top: .6rem;">Forward‑Delivery Risk Studio</h1>
      <p class="small-muted">
        An open, educational front‑end for exploring pre‑payment exposure in card transactions.
        This interface helps risk and operations teams reason about booking‑level risk, policy levers,
        and explainable outcomes — all with synthetic data for demonstration purposes.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Note: This is an educational demo using synthetic data. No advertising or merchandise is presented here.")

# --- Hero images
cols = st.columns([1,1])
with cols[0]:
    st.image(ASSETS / "payment_rails.png", caption="Conceptual view of payment rails & participants", use_column_width=True)
with cols[1]:
    st.image(ASSETS / "horizon_heatmap.png", caption="Conceptual risk vs. time‑to‑service (illustrative)", use_column_width=True)

st.markdown("### Why this matters to the payments industry")
st.markdown(
    """
- **Pre‑payment exposure** appears when end‑customers pay before service delivery (e.g., travel, tickets, courses, hotel bookings).
- If delivery fails, **dispute liability** can fall back to the acquirer/PSP, so risk must be quantified early.
- **Static reserves and flat settlement delays** are blunt tools — they can penalize all merchants equally and still miss emerging stress.
- A **booking‑level view** and **explainable actioning** (reserve %, payout delay) help reduce loss *and* improve fairness.
"""
)

st.markdown("### What you can do in this demo")
st.markdown(
    """
- 📁 **Explore synthetic data**: inspect bookings, merchants, and risk signals.
- 🧠 **Score & explain**: compute a booking‑level score and view key drivers.
- ⚖️ **Simulate policies**: compare reserve/delay settings against expected loss.
- 📒 **Review governance**: see documentation, data boundaries, and ethics notes.
"""
)

# Quick nav (available in newer Streamlit versions)
try:
    st.markdown("#### Jump to a section")
    st.page_link("pages/1_📊_Dataset_Explorer.py", label="Dataset Explorer")
    st.page_link("pages/2_🧠_Scoring_Demo.py", label="Scoring Demo")
    st.page_link("pages/3_⚖️_Policy_Simulator.py", label="Policy Simulator")
    st.page_link("pages/4_🔍_Explainability.py", label="Explainability")
    st.page_link("pages/5_📒_Ethics_&_Governance.py", label="Ethics & Governance")
except Exception:
    st.info("Use the sidebar to switch pages.")

st.markdown("### How to read this landing page")
with st.expander("Industry context (concise)"):
    st.write(
        """
**Forward‑delivery risk** refers to potential loss when a consumer prepays, but the merchant later cannot or does not deliver.
It is distinct from stolen‑card fraud: the cardholder is legitimate, yet service fails. As a result, risk managers look at:
- **Service horizon** (days until delivery)
- **Merchant health** (e.g., recent performance and operations signals)
- **Policies in place** (rolling reserve %, settlement delay, deposit rules)

The rest of this app uses synthetic data to visualize these factors and the impact of different policy levers.
        """
    )

with st.expander("Transparency & boundaries"):
    st.write(
        """
- This demo is **not** legal, regulatory, or underwriting advice.
- Data is entirely **synthetic**; any resemblance to real entities is coincidental.
- Outputs are provided for educational & prototyping purposes only.
        """
    )
