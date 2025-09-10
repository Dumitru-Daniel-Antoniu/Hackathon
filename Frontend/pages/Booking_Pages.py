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
      .stButton > button[kind="primary"] { border-radius: 12px; box-shadow: 0 8px 18px rgba(11,19,43,0.12); }
      .tbl-header { font-weight: 600; font-size: 0.95rem; opacity: 0.95; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Disable Enter submit globally
st.markdown(
    """
    <style>.stApp { overflow: auto !important; } .block-container { padding-top: 3rem !important; }</style>
    <script>
      window.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          const el = document.activeElement;
          if (!el) return;
          const tag = (el.tagName || '').toLowerCase();
          if (tag === 'input' || tag === 'textarea') {
            e.preventDefault(); e.stopImmediatePropagation(); e.stopPropagation(); return false;
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
# Session state & preset/default values
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "company_name": "",
    "typical_lead_time": "",  # 0–120
    "pfr_percent": "",  # 0.00–1.00
    "shock_flag": False,
    "days_in_advance": "",  # 0–420
    "booking_amount": "",  # 0–10000
}
PRESET_DEFAULTS: Dict[str, Any] = {
    "company_name": "Wizz Air",
    "typical_lead_time": "30",
    "pfr_percent": "0.05",
    "shock_flag": False,
    "days_in_advance": "60",
    "booking_amount": "500",
}

if "initialized" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["initialized"] = True

# Keep results but only show metrics on Send run
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

# Flags to safely mutate inputs with buttons placed *after* widgets
if "apply_defaults" not in st.session_state:
    st.session_state["apply_defaults"] = False
if "apply_clear" not in st.session_state:
    st.session_state["apply_clear"] = False

# If a flag is set from a previous click, apply BEFORE inputs are created, then unset.
if st.session_state.get("apply_defaults"):
    for k, v in PRESET_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["apply_defaults"] = False
if st.session_state.get("apply_clear"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["apply_clear"] = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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

    return {
        "typical_horizon": tlt,
        "base_fdr": pfr,
        "shock_flag": shock,
        "days_in_advance": dia,
        "booking_amount": amt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Header + Inputs (fields first)
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

row_cols = st.columns([1.2, 0.9, 1.1, 0.7, 0.9, 0.9], gap="small")
with row_cols[0]:
    st.text_input("Company name", key="company_name", placeholder="e.g., Wizz Air", label_visibility="collapsed")
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
# Buttons BELOW fields: [Set default] [tiny gap] [Send] ---spacer--- [Clear]
# (Set default sits just left of Send, with a small spacer)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Buttons BELOW fields: [Set default] [tiny gap] [Send] ---spacer--- [Clear]
# ──────────────────────────────────────────────────────────────────────────────
btn_cols = st.columns([1, 0.01, 1, 6.94, 1], gap="small")  # moved Send slightly left by shrinking spacer
with btn_cols[0]:
    send_clicked = st.button("Send", type="primary", key="send_btn")
with btn_cols[2]:
    defaults_clicked = st.button("Set default", key="defaults_btn")
with btn_cols[4]:
    clear_clicked = st.button("Clear", key="clear_btn")

# Handle Set default / Clear via flags then rerun to safely prefill/clear inputs
if defaults_clicked:
    st.session_state["apply_defaults"] = True
    st.rerun()

if clear_clicked:
    st.session_state["apply_clear"] = True
    st.session_state["last_result"] = None
    st.rerun()

# Handle Send (show metrics only on this run)
send_success = False
if send_clicked:
    st.session_state["last_error"] = None
    payload = validate_and_payload()
    if payload is not None:
        try:
            with st.spinner("Scoring…"):
                resp = requests.post(BACKEND_URL, json=payload, timeout=30)
            if resp.status_code >= 400:
                st.session_state["last_error"] = f"Backend returned HTTP {resp.status_code}: {resp.text[:500]}"
                st.session_state["last_result"] = None
            else:
                data = resp.json()
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
                send_success = True
        except Exception as e:
            st.session_state["last_error"] = f"Request failed: {e}"
            st.session_state["last_result"] = None

# ──────────────────────────────────────────────────────────────────────────────
# Response area: ONLY metrics, one line, only on the run where Send was clicked
# ──────────────────────────────────────────────────────────────────────────────
if send_success and st.session_state.get("last_result") is not None:
    res = st.session_state["last_result"]
    metrics = []

    if res.get("risk_score") is not None:
        try:
            metrics.append(("Risk score", f"{float(res['risk_score']):.2f}"))
        except:
            pass
    if res.get("risk_score") is not None:
        try:
            metrics.append(("Chance of non-delivery", f"{float(res['risk_score']):.2%}"))
        except:
            pass
    if res.get("risk_tier"):
        metrics.append(("Risk tier", str(res["risk_tier"])))
    if res.get("suggested_reserve_percent") is not None:
        try:
            metrics.append(("Suggested reserve (%)", f"{float(res['suggested_reserve_percent']):.2f}"))
        except:
            pass
    if res.get("suggested_settlement_delay_days") is not None:
        try:
            metrics.append(("Suggested payout delay (days)", f"{float(res['suggested_settlement_delay_days']):.0f}"))
        except:
            pass

    if metrics:
        cols = st.columns(len(metrics), gap="small")
        for c, (label, value) in zip(cols, metrics):
            with c:
                st.metric(label, value)
