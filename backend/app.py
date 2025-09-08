# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any
import csv, io, uuid

app = FastAPI(title="Policy Simulator API", version="1.0.0")

# Optional CORS (tweak for your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- In-memory store for uploaded datasets ---
_DATASETS: Dict[str, List[Dict[str, Any]]] = {}

def _to_number_if_possible(v: str):
    """Attempt to cast CSV strings to float/int when appropriate."""
    if v is None:
        return v
    s = v.strip()
    if s == "":
        return None
    try:
        # int when exact, else float
        f = float(s.replace(",", ""))  # accept 1,234.56
        i = int(f)
        return i if f == i else f
    except Exception:
        return s

@app.get("/health")
def health():
    return {"status": "ok"}

# ----- CSV Upload -----
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), delimiter: Optional[str] = None):
    """
    Upload a CSV (multipart/form-data) and get back a dataset_id for later use.
    Automatically type-coerces numeric-looking fields.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    text = raw.decode("utf-8-sig")  # handle BOM if present
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(text[:2048])
        detected_delim = dialect.delimiter
    except Exception:
        detected_delim = ","
    use_delim = delimiter or detected_delim

    reader = csv.DictReader(io.StringIO(text), delimiter=use_delim)
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must have a header row")

    rows: List[Dict[str, Any]] = []
    for r in reader:
        typed = {k: _to_number_if_possible(v) for k, v in r.items()}
        rows.append(typed)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV appears to be empty")

    dataset_id = str(uuid.uuid4())
    _DATASETS[dataset_id] = rows
    return {
        "dataset_id": dataset_id,
        "rows": len(rows),
        "columns": reader.fieldnames,
        "delimiter_used": use_delim,
        "filename": file.filename,
    }

# ----- Scoring -----
class ScoreRequest(BaseModel):
    item: Optional[Dict[str, float]] = None
    items: Optional[List[Dict[str, float]]] = None
    weights: Optional[Dict[str, float]] = None  # if None => equal weights
    scale_to: float = Field(100.0, gt=0, description="Scale scores to this max (default 100)")

def _score_one(obj: Dict[str, float], weights: Optional[Dict[str, float]], scale_to: float) -> float:
    feats = obj.keys() if weights is None else (k for k in obj.keys() if k in weights)
    vals = []
    wts = []
    for k in feats:
        v = obj[k]
        if isinstance(v, (int, float)):
            w = 1.0 if (weights is None) else float(weights.get(k, 0.0))
            vals.append(float(v) * w)
            wts.append(abs(w))
    if not wts or sum(wts) == 0:
        return 0.0
    raw = sum(vals) / sum(wts)   # weighted average
    return float(raw) / 1.0 * scale_to

@app.post("/score")
def score(req: ScoreRequest):
    if not req.item and not req.items:
        raise HTTPException(status_code=400, detail="Provide 'item' or 'items'")
    if req.item and req.items:
        raise HTTPException(status_code=400, detail="Provide only one of 'item' or 'items'")

    if req.item:
        s = _score_one(req.item, req.weights, req.scale_to)
        return {"score": s}

    scores = [_score_one(it, req.weights, req.scale_to) for it in req.items or []]
    return {"scores": scores, "count": len(scores)}

# ----- Policy Simulation -----
class PolicyRequest(BaseModel):
    dataset_id: str
    target_column: str
    policy: Literal["tax", "bonus", "cap", "threshold"] = "tax"
    value: float = Field(..., description="Meaning depends on policy")

@app.post("/simulate_policy")
def simulate_policy(req: PolicyRequest):
    """
    Apply a simple policy to a numeric column and report impact.
    Policies:
      - tax:       new = old * (1 - value)         (value as fraction, e.g., 0.1 = 10%)
      - bonus:     new = old + value               (value as absolute increment)
      - cap:       new = min(old, value)           (value is the cap)
      - threshold: keep only rows with old >= value (counts only)
    """
    if req.dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="Unknown dataset_id")
    rows = _DATASETS[req.dataset_id]

    # Validate column
    if not rows or req.target_column not in rows[0]:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found")

    # Extract numeric values
    vals: List[float] = []
    for r in rows:
        v = r.get(req.target_column)
        if isinstance(v, (int, float)):
            vals.append(float(v))
        else:
            # Skip non-numeric/blank
            continue

    if not vals:
        raise HTTPException(status_code=400, detail=f"No numeric values in '{req.target_column}'")

    import math
    n = len(vals)
    total_before = float(sum(vals))
    min_before, max_before = float(min(vals)), float(max(vals))
    avg_before = total_before / n

    # Apply policy
    if req.policy == "tax":
        after_vals = [x * (1.0 - req.value) for x in vals]
    elif req.policy == "bonus":
        after_vals = [x + req.value for x in vals]
    elif req.policy == "cap":
        after_vals = [min(x, req.value) for x in vals]
    elif req.policy == "threshold":
        kept = [x for x in vals if x >= req.value]
        return {
            "policy": req.policy,
            "value": req.value,
            "count_before": n,
            "count_after": len(kept),
            "kept_ratio": (len(kept) / n) if n else 0.0,
            "threshold": req.value,
            "min_kept": float(min(kept)) if kept else None,
            "max_kept": float(max(kept)) if kept else None,
            "avg_kept": (float(sum(kept)) / len(kept)) if kept else None,
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported policy")

    total_after = float(sum(after_vals))
    delta = total_after - total_before
    return {
        "policy": req.policy,
        "value": req.value,
        "count": n,
        "total_before": total_before,
        "total_after": total_after,
        "delta": delta,
        "delta_pct": (delta / total_before) if total_before != 0 else None,
        "min_before": min_before,
        "max_before": max_before,
        "avg_before": avg_before,
        "min_after": float(min(after_vals)),
        "max_after": float(max(after_vals)),
        "avg_after": float(total_after / n),
        "preview_after_first5": after_vals[:5],
    }
