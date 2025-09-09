from __future__ import annotations
import pandas as pd
from ml.artifacts import get_artifacts
from services.colleague_adapter import predict_proba_colleague
from services.policy import _policy

# Simple errors used by routes
class RowValidationError(Exception): ...
class CsvParseError(Exception): ...

def _to_records_df(payloads):

    # Accept dict / list[dict] / DataFrame; reject naked arrays
    if isinstance(payloads, pd.DataFrame):
        df = payloads.copy()
    elif isinstance(payloads, dict):
        df = pd.DataFrame([payloads])
    elif isinstance(payloads, list):
        if all(isinstance(x, dict) for x in payloads):
            df = pd.DataFrame(payloads)
        else:
            raise RowValidationError("Body must be an object or list of objects with named fields.")
    else:
        raise RowValidationError("Unsupported payload type.")
    df.columns = [str(c) for c in df.columns]
    return df

def score_items(payloads: list[dict]) -> list[dict]:
    art = get_artifacts()
    model = art["model"]
    feature_list = art.get("input_cols")
    classes = art.get("classes")

    df_raw = _to_records_df(payloads)

    probs = predict_proba_colleague(model, df_raw, feature_list, classes)

    out = []
    for row, p in zip(df_raw.to_dict(orient="records"), probs):
        prob = float(p)
        days = float(row.get("DAYS_IN_ADVANCE") or row.get("days_in_advance") or 0.0)
        amount = float(row.get("BOOKING_AMOUNT") or row.get("booking_amount") or 0.0)
        reserve_pct, delay_days, tier = _policy(prob, days)
        out.append({
            "probability": prob*100,
            "risk_score": prob,
            "risk_tier": tier,
            "suggested_reserve_percent": reserve_pct,
            "suggested_settlement_delay_days": delay_days,
            "merchant_id": row.get("MERCHANT_ID") or row.get("merchant_id"),
            "vertical": row.get("VERTICAL") or row.get("vertical"),
            "country": row.get("COUNTRY") or row.get("country"),
            "days_in_advance": days,
            "booking_amount": amount,
        })
    return out

# CSV helper (unchanged idea, but now it flows into the colleague path)
import io
def score_csv_bytes(csv_bytes: bytes) -> list[dict]:
    if not csv_bytes:
        raise CsvParseError("Empty file.")
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), sep=None, engine="python", dtype=str)
    except Exception as e:
        raise CsvParseError(f"Could not parse CSV: {e!r}")
    df.columns = [c.strip() for c in df.columns]
    return score_items(df)
