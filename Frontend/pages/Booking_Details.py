import streamlit as st
from typing import Dict, Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Score for an individual booking", page_icon="🧮", layout="wide")

# Minimal styles + disable Enter behavior
st.markdown(
    """
    <style>
      .stApp { overflow: auto !important; }
      .block-container { padding-top: 3rem !important; }

      .tbl-header { font-weight: 600; font-size: 0.95rem; opacity: 0.95; }
      .grid-6 { display: grid; grid-template-columns: 1.2fr 0.9fr 1.1fr 0.7fr 0.9fr 0.9fr; gap: .6rem; align-items: center; }

      .stButton > button[kind="primary"] {
        border-radius: 12px;
        box-shadow: 0 8px 18px rgba(11,19,43,0.12);
      }
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
        e.preventDefault();
        e.stopImmediatePropagation();
        e.stopPropagation();
        return false;
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
    "typical_lead_time": "",
    "pfr_percent": "",
    "shock_flag": False,
    "days_in_advance": "",
    "booking_amount": "",
}
if "initialized" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["initialized"] = True
if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None

def clear_only_fields():
    """Reset input fields only; keep last_payload intact."""
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
    st.text_input("Company name", key="company_name", placeholder="e.g., Acme Travel Ltd.", label_visibility="collapsed")

with row_cols[1]:
    st.text_input("Typical lead time", key="typical_lead_time", placeholder="0–120", label_visibility="collapsed")

with row_cols[2]:
    st.text_input("Prior forward-delivery risk percent", key="pfr_percent", placeholder="0.00–1.00", label_visibility="collapsed")

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
    try: return int(s.replace(",", "").replace(" ", ""))
    except: return None

def _to_float(s: str) -> Optional[float]:
    if not s: return None
    try: return float(s.replace(",", "").replace(" ", ""))
    except: return None

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
    if not (0 <= tlt <= 120):
        st.toast("Typical lead time must be between 0 and 120.", icon="⚠️"); return None
    if not (0.0 <= pfr <= 1.0):
        st.toast("Prior forward-delivery risk percent must be between 0.00 and 1.00.", icon="⚠️"); return None
    if not (0 <= dia <= 420):
        st.toast("Days in advance must be between 0 and 420.", icon="⚠️"); return None
    if not (0 <= amt <= 10000):
        st.toast("Booking amount must be between 0 and 10000.", icon="⚠️"); return None

    return {
        "record": {
            "Company name": name,
            "Typical lead time": tlt,
            "Prior forward-delivery risk percent": pfr,
            "Shock flag": shock,
            "Days in advance": dia,
            "Booking amount": amt,
        }
    }

# ──────────────────────────────────────────────────────────────────────────────
# Buttons (Send left, Clear right)
# ──────────────────────────────────────────────────────────────────────────────
send_col, spacer, clear_col = st.columns([1, 7, 1])
with send_col:
    if st.button("Send", type="primary", key="send_btn"):
        payload = validate_and_payload()
        if payload is not None:
            st.session_state["last_payload"] = payload  # no success pop-up

with clear_col:
    st.button("Clear", key="clear_btn", on_click=clear_only_fields)

# ──────────────────────────────────────────────────────────────────────────────
# Response section (separate from inputs; shown only if payload exists)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state["last_payload"] is not None:
    st.markdown("**Request preview**")
    st.json(st.session_state["last_payload"])
