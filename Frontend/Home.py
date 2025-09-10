# app.py — Home
import streamlit as st

st.set_page_config(
    page_title="PrePayGuard",
    page_icon="💳",
    layout="wide",
)

# ------------------------- THEME & STYLES -------------------------
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

# =========================
# HERO
# =========================
with st.container():
    st.markdown(
        """
        <div class="hero">
          <h1>PrePayGuard</h1>
          <p class="subtle">
            An interactive workspace for <span class="glow">forward-delivery risk</span>.  
            Quantify exposure when bookings are prepaid, and explore explainable controls like reserves and settlement delays—designed for acquirers, PSPs, and risk teams.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

# =========================
# IMAGES SECTION
# =========================
with st.container():
    st.subheader("Core capabilities at a glance")
    st.caption("Three pillars of the product experience—aligned to how data moves through your risk pipeline.")

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div class="svg-card">
            <!-- bookings image -->
            <svg viewBox="0 0 200 200" width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
              <defs><linearGradient id="grad1" x1="0" x2="1">
                <stop offset="0%" stop-color="#19c6d1"/><stop offset="100%" stop-color="#7ae2f2"/>
              </linearGradient></defs>
              <rect x="20" y="30" width="160" height="140" rx="12" fill="#141f33" stroke="rgba(255,255,255,.15)"/>
              <rect x="35" y="50" width="130" height="20" rx="4" fill="url(#grad1)"/>
              <rect x="35" y="80" width="110" height="15" rx="4" fill="#1f2e4a"/>
              <rect x="35" y="105" width="90" height="15" rx="4" fill="#1f2e4a"/>
              <rect x="35" y="130" width="70" height="15" rx="4" fill="#1f2e4a"/>
            </svg>
            <div class="subtle img-caption">📑 <b>Bookings:</b> schemas and input validation for scoring pipelines</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="svg-card">
            <!-- scoring image -->
            <svg viewBox="0 0 200 200" width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
              <defs><linearGradient id="grad2" x1="0" x2="1">
                <stop offset="0%" stop-color="#19c6d1"/><stop offset="100%" stop-color="#7ae2f2"/>
              </linearGradient></defs>
              <circle cx="100" cy="100" r="70" fill="#141f33" stroke="url(#grad2)" stroke-width="4"/>
              <text x="100" y="108" text-anchor="middle" font-size="26" fill="white">Risk</text>
              <rect x="50" y="150" width="100" height="15" rx="4" fill="url(#grad2)"/>
            </svg>
            <div class="subtle img-caption">🎯 <b>Scoring:</b> model probabilities, risk tiers & policy recommendations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="svg-card">
            <!-- policy/features image -->
            <svg viewBox="0 0 200 200" width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
              <defs><linearGradient id="grad3" x1="0" x2="1">
                <stop offset="0%" stop-color="#19c6d1"/><stop offset="100%" stop-color="#7ae2f2"/>
              </linearGradient></defs>
              <polygon points="100,20 180,60 160,180 40,180 20,60" fill="#141f33" stroke="url(#grad3)" stroke-width="3"/>
              <rect x="60" y="80" width="80" height="15" rx="4" fill="url(#grad3)"/>
              <rect x="60" y="105" width="60" height="12" rx="4" fill="#1f2e4a"/>
              <rect x="60" y="125" width="50" height="12" rx="4" fill="#1f2e4a"/>
            </svg>
            <div class="subtle img-caption">⚙️ <b>Features & Policy:</b> engineered metrics, model prep, reserve & delay logic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

# =========================
# DESCRIPTION SECTION
# =========================
with st.container():
    st.subheader("How it fits together")
    st.markdown(
        """
- **Data entry**: A base dataset or an user defined dataset is used to analyze the scoring and policy. Alternatively, a single-booking form captures the variables your risk models need (e.g., days-in-advance, booking amount, prior forward-delivery risk percent). Inputs remain human-friendly while being validated at the edge.
- **Scoring & policy**: The scoring service returns a probability and human-readable **risk tier**, alongside suggested **reserve percent** and **settlement delay**. These suggestions align with how acquirers and PSPs mitigate prepayment exposure in practice.
- **Portfolio views**: Aggregation pages summarize where **exposure** and **estimated loss** concentrate across verticals and merchants, helping teams prioritize outreach and policy changes where they matter most.
        """
    )

# ↓↓↓ slightly less space before Why section ↓↓↓
st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

# =========================
# WHY THIS MATTERS SECTION
# =========================
with st.container():
    st.subheader("Why this matters to the payments industry")
    st.markdown(
        """
- **Forward-delivery risk is systemic**: Any business taking payment **before** delivering service (airlines, events, tours, hotels, transport) introduces potential non-delivery risk that can push liability back to the acquirer/PSP.
- **Precision over blanket policies**: Booking-level scoring lets you replace broad, static controls with **targeted reserves and payout timing**, improving protection without penalizing high-quality merchants.
- **Operational clarity**: Outputs are expressed in business language—**probability**, **tier**, **reserve%**, **delay (days)**—so risk ops, merchant ops, and commercial teams can act quickly and consistently.
        """
    )
