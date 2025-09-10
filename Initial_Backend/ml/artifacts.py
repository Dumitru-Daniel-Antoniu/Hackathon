import os, json
from pathlib import Path
import joblib
import xgboost as xgb

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts_xgb"))
PKL_PATH     = ARTIFACT_DIR / "xgb_best_model.pkl"

def get_artifacts():

    if PKL_PATH.exists():
        model = joblib.load(PKL_PATH)
        # Patch quirks for old pickles
        for k, v in {"use_label_encoder": False, "gpu_id": None, "predictor": None}.items():
            if not hasattr(model, k):
                setattr(model, k, v)
    else:
        raise FileNotFoundError(f"No model found at {PKL_PATH}")

    return {
        "artifact_dir": ARTIFACT_DIR,
        "model": model,
        "preprocess": None,     # <- important: disable old OHE path
    }
