import os, json
from pathlib import Path
import joblib
import xgboost as xgb

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts_xgb"))
PKL_PATH     = ARTIFACT_DIR / "xgb_best_model.pkl"
JSON_PATH    = ARTIFACT_DIR / "xgb_model.json"
FEATURES_PATH= ARTIFACT_DIR / "feature_list.json"
CLASSES_PATH = ARTIFACT_DIR / "label_classes.json"

def get_artifacts():
    # Load model (prefer JSON booster if present)
    if JSON_PATH.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(JSON_PATH))
    elif PKL_PATH.exists():
        model = joblib.load(PKL_PATH)
        # Patch quirks for old pickles
        for k, v in {"use_label_encoder": False, "gpu_id": None, "predictor": None}.items():
            if not hasattr(model, k):
                setattr(model, k, v)
    else:
        raise FileNotFoundError(f"No model found at {JSON_PATH} or {PKL_PATH}")

    # Feature order (optional but strongly recommended)
    feature_list = None
    if FEATURES_PATH.exists():
        feature_list = json.loads(FEATURES_PATH.read_text())

    # Label classes (for deterministic encodings)
    classes = {}
    if CLASSES_PATH.exists():
        classes = json.loads(CLASSES_PATH.read_text())

    return {
        "artifact_dir": ARTIFACT_DIR,
        "model": model,
        "preprocess": None,     # <- important: disable old OHE path
        "input_cols": feature_list,
        "classes": classes
    }
