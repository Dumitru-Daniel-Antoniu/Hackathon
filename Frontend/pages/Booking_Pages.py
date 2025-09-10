import streamlit as st
from typing import Dict, Any, Optional
import requests

BACKEND_URL = "http://localhost:8000/score/individual/simple"

st.set_page_config(page_title="Score for an individual booking", page_icon="💳", layout="wide")

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

# Minimal styles + disable Enter behavior
st.markdown(
    """
    <style>
      .stApp { overflow: auto !important; }
      .block-container { padding-top: 3rem !important; }
      .tbl-header { font-weight: 600; font-size: 0.95rem; opacity: 0.95; }
      .grid-6 { display: grid; grid-template-columns: 1.2fr 0.9fr 1.1fr 0.7fr 0.9fr 0.9fr; gap: .6rem; align-items: center; }
      .stButton > button[kind="primary"] { border-radius: 12px; box-shadow: 0 8px 18px rgba(11,19,43,0.12); }
      .result-card { margin-top: 0.5rem; padding: .6rem .8rem; border-radius: 10px; border:1px dashed rgba(0,0,0,.08); }
    </style>
    <script>
      // Block Enter globally so it never applies or triggers reruns
      window.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          const el = document.activeElement;
          if (!el) return;
          const tag = (el.tagName || '').toLowerCase();
          if (tag === 'input' || tag === 'textarea') {
            e.preventDefault();
            e.stopImmediatePropagation();
            e.stopPropagation();
            return false;
          }
        }
      }, true);
      window.addEventListener('submit', function(e) {
        e.preventDefault(); e.stopImmediatePropagation(); e.stopPropagation(); return false;
      }, true);
    </script>
    """,
    unsafe_allow_html=True,
)

st.title("Score for an individual booking")

# ──────────────────────────────────────────────────────────────────────────────
# Session state: keep raw text exactly; parse only on Send; preserve response
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "company_name": "",
    "typical_lead_time": "",  # 0–120
    "pfr_percent": "",  # 0.00–1.00
    "shock_flag": False,
    "days_in_advance": "",  # 0–420
    "booking_amount": "",  # 0–10000
}
if "initialized" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["initialized"] = True
if "last_request" not in st.session_state:
    st.session_state["last_request"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


def clear_only_fields():
    """Reset input fields only; keep last_result and last_request intact."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
# Header row
# ──────────────────────────────────────────────────────────────────────────────
hdr = st.columns([1.2, 0.9, 1.1, 0.7, 0.9, 0.9], gap="small")
labels = [
    "Company name",
    "Typical lead time",
    "Prior forward-delivery risk percent",
    "Shock flag",
    "Days in advance",
    "Booking amount",
]
for i, lab in enumerate(labels):
    with hdr[i]:
        st.markdown(f'<div class="tbl-header">{lab}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Single-row inputs
# ──────────────────────────────────────────────────────────────────────────────
row_cols = st.columns([1.2, 0.9, 1.1, 0.7, 0.9, 0.9], gap="small")

with row_cols[0]:
    st.text_input("Company name", key="company_name", placeholder="e.g., Acme Travel Ltd.",
                  label_visibility="collapsed")

with row_cols[1]:
    st.text_input("Typical lead time", key="typical_lead_time", placeholder="0–120", label_visibility="collapsed")

with row_cols[2]:
    st.text_input("Prior forward-delivery risk percent", key="pfr_percent", placeholder="0.00–1.00",
                  label_visibility="collapsed")

with row_cols[3]:
    st.checkbox("Shock flag", key="shock_flag", label_visibility="collapsed")

with row_cols[4]:
    st.text_input("Days in advance", key="days_in_advance", placeholder="0–420", label_visibility="collapsed")

with row_cols[5]:
    st.text_input("Booking amount", key="booking_amount", placeholder="0–10000", label_visibility="collapsed")


# ──────────────────────────────────────────────────────────────────────────────
# Validation & payload building
# ──────────────────────────────────────────────────────────────────────────────
def _to_int(s: str) -> Optional[int]:
    if not s: return None
    try:
        return int(s.replace(",", "").replace(" ", ""))
    except:
        return None


def _to_float(s: str) -> Optional[float]:
    if not s: return None
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except:
        return None


def validate_and_payload() -> Optional[Dict[str, Any]]:
    name = st.session_state["company_name"].strip()
    tlt = _to_int(st.session_state["typical_lead_time"])
    pfr = _to_float(st.session_state["pfr_percent"])
    dia = _to_int(st.session_state["days_in_advance"])
    amt = _to_float(st.session_state["booking_amount"])
    shock = bool(st.session_state["shock_flag"])

    if name == "" or tlt is None or pfr is None or dia is None or amt is None:
        st.toast("All fields must be filled with numeric values in the shown ranges.", icon="⚠️")
        return None
    if not (0 <= tlt <= 120):   st.toast("Typical lead time must be between 0 and 120.", icon="⚠️"); return None
    if not (0.0 <= pfr <= 1.0): st.toast("Prior forward-delivery risk percent must be between 0.00 and 1.00.",
                                         icon="⚠️"); return None
    if not (0 <= dia <= 420):   st.toast("Days in advance must be between 0 and 420.", icon="⚠️"); return None
    if not (0 <= amt <= 10000): st.toast("Booking amount must be between 0 and 10000.", icon="⚠️"); return None

    # The backend endpoint /score/individual/simple expects a single object.
    # We send keys matching our form field names.
    payload = {
        "typical_horizon": tlt,
        "base_fdr": pfr,
        "shock_flag": shock,
        "days_in_advance": dia,
        "booking_amount": amt,
    }
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Buttons (Send left, Clear right)
# ──────────────────────────────────────────────────────────────────────────────
send_col, spacer, clear_col = st.columns([1, 7, 1])

with send_col:
    if st.button("Send", type="primary", key="send_btn"):
        st.session_state["last_error"] = None
        payload = validate_and_payload()
        if payload is not None:
            # Store request preview
            st.session_state["last_request"] = payload

            # Call backend
            try:
                with st.spinner("Scoring…"):
                    resp = requests.post(BACKEND_URL, json=payload, timeout=30)
                if resp.status_code >= 400:
                    st.session_state["last_error"] = f"Backend returned HTTP {resp.status_code}: {resp.text[:500]}"
                    st.session_state["last_result"] = None
                else:
                    # Expect a single result dict with keys like in your ScoreOutWithContext
                    data = resp.json()
                    # Some backends might return {"result": {...}} or a list—handle a few cases gracefully:
                    if isinstance(data, dict) and "result" in data:
                        result = data["result"]
                    elif isinstance(data, list):
                        result = data[0] if data else {}
                    else:
                        result = data

                    st.session_state["last_result"] = {
                        "probability": result.get("probability"),
                        "risk_score": result.get("risk_score"),
                        "risk_tier": result.get("risk_tier"),
                        "suggested_reserve_percent": result.get("suggested_reserve_percent"),
                        "suggested_settlement_delay_days": result.get("suggested_settlement_delay_days"),
                        "merchant_id": result.get("merchant_id"),
                        "vertical": result.get("vertical"),
                        "country": result.get("country"),
                        "days_in_advance": result.get("days_in_advance"),
                        "booking_amount": result.get("booking_amount"),
                    }
            except Exception as e:
                st.session_state["last_error"] = f"Request failed: {e}"
                st.session_state["last_result"] = None

with clear_col:
    st.button("Clear", key="clear_btn", on_click=clear_only_fields)

# ──────────────────────────────────────────────────────────────────────────────
# Response area (separate from inputs)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

if st.session_state.get("last_request") is not None:
    st.markdown("**Request preview**")
    st.json(st.session_state["last_request"])

if st.session_state.get("last_result") is not None:
    st.markdown("**Model result**")
    res = st.session_state["last_result"]

    # Friendly KPI row if the expected keys are present
    r1, r2, r3 = st.columns(3)
    if res.get("risk_score") is not None:
        r1.metric("Risk score", f"{float(res['risk_score']):.2f}")
    if res.get("risk_score") is not None:
        r2.metric("Chance of non-delivery", f"{float(res['risk_score']):.2%}")
    if res.get("risk_tier"):
        r3.metric("Risk tier", str(res["risk_tier"]))

    # Policy hints if available
    if (res.get("suggested_reserve_percent") is not None) or (res.get("suggested_settlement_delay_days") is not None):
        p1, p2 = st.columns(2)
        if res.get("suggested_reserve_percent") is not None:
            p1.metric("Suggested reserve (%)", f"{float(res['suggested_reserve_percent']):.2f}")
        if res.get("suggested_settlement_delay_days") is not None:
            p2.metric("Suggested payout delay (days)", f"{float(res['suggested_settlement_delay_days']):.0f}")

    # Raw JSON for completeness
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.json(res)
    st.markdown('</div>', unsafe_allow_html=True)
