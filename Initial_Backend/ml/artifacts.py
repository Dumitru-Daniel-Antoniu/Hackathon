import os
from pathlib import Path

import joblib
from functools import lru_cache

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts_xgb"))
PREPROCESS_PATH = ARTIFACT_DIR / "preprocess_ohe.pkl"
MODEL_PATH = ARTIFACT_DIR / "booking_risk_xgb_model.pkl"

def _patch_xgb_attrs(xgb):
    defaults = {
        "use_label_encoder": False,
        "gpu_id": None,
        "predictor": None,
    }
    patched = []
    for k, v in defaults.items():
        if not hasattr(xgb, k):
            try:
                setattr(xgb, k, v)
                patched.append(k)
            except Exception:
                pass
    if patched:
        print(f"[xgb-patch] added missing attrs: {patched}")
    return xgb

def _fallback_input_cols():
    return [
        # numeric base
        "TRUST_SCORE","PRIOR_CB_RATE","REFUND_RATE","CANCEL_RATE",
        "SENTIMENT","SALES_GROWTH_3M","PAYOUT_DELAY_DAYS","RESERVE_PERCENT",
        "DEPOSIT_POLICY_PERCENT","DAYS_IN_ADVANCE","BOOKING_AMOUNT",
        "NEW_MERCHANT","SHOCK_FLAG",
        # engineered numeric
        "refund_cancel_ratio","shock_adjusted_lead","merchant_stability","high_risk_vertical_flag",
        # categorical base + engineered
        "VERTICAL","COUNTRY","days_in_advance_bucket"
    ]

@lru_cache(maxsize=1)
def get_artifacts():
    preprocess = joblib.load(PREPROCESS_PATH)   # fitted ColumnTransformer
    model = joblib.load(MODEL_PATH)             # fitted XGBClassifier
    model = _patch_xgb_attrs(model)

    try:
        input_cols = list(preprocess.feature_names_in_)
    except Exception:
        input_cols = _fallback_input_cols()

    return {
        "preprocess": preprocess,
        "model": model,
        "input_cols": input_cols,
    }
