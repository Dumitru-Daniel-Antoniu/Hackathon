from typing import List, Iterable
import io
import pandas as pd
import numpy as np

from schemas.bookings import BookingIn
from schemas.scoring import ScoreOutWithContext
from services.feature_engineering import add_engineered_features
from ml.artifacts import get_artifacts
from services.policy import post_policy

class RowValidationError(Exception): ...
class CsvParseError(Exception): ...

def _rows_to_booking(items: Iterable[dict]) -> List[BookingIn]:
    out: List[BookingIn] = []
    for row in items:
        try:
            # convert NaN -> None so pydantic is happy
            clean = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            out.append(BookingIn.model_validate(clean))
        except Exception as e:
            raise RowValidationError(str(e))
    return out

def score_items(items: List[BookingIn]) -> List[ScoreOutWithContext]:
    if not items:
        return []
    artifacts = get_artifacts()
    preprocess = artifacts["preprocess"]
    model = artifacts["model"]
    input_cols = artifacts["input_cols"]

    df = pd.DataFrame([it.model_dump() for it in items])
    df = add_engineered_features(df)

    for col in input_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = preprocess.transform(df[input_cols])
    probs = model.predict_proba(X)[:, 1]

    results: List[ScoreOutWithContext] = []
    for i, p in enumerate(probs):
        b = items[i]
        tier, reserve, delay = post_policy(float(p), float(b.DAYS_IN_ADVANCE or 0))
        results.append(ScoreOutWithContext(
            probability=float(p),
            risk_score=round(float(p) * 100, 2),  # or keep raw if you prefer
            risk_tier=tier,
            suggested_reserve_percent=reserve,
            suggested_settlement_delay_days=delay,
            merchant_id=b.MERCHANT_ID,
            vertical=b.VERTICAL,
            country=b.COUNTRY,
            days_in_advance=b.DAYS_IN_ADVANCE,
            booking_amount=b.BOOKING_AMOUNT,
        ))
    return results

def score_csv_bytes(data: bytes, encoding: str = "utf-8-sig") -> List[ScoreOutWithContext]:
    try:
        df = pd.read_csv(io.BytesIO(data), encoding=encoding)
    except Exception as e:
        raise CsvParseError(str(e))
    if df.empty:
        raise CsvParseError("CSV contains no rows")
    items = _rows_to_booking(df.to_dict(orient="records"))
    return score_items(items)

# Optional: streaming for large CSVs
def score_csv_stream(data: bytes, chunksize: int = 10_000, encoding: str = "utf-8-sig") -> List[ScoreOutWithContext]:
    try:
        buf = io.BytesIO(data)
        reader = pd.read_csv(buf, encoding=encoding, chunksize=chunksize)
    except Exception as e:
        raise CsvParseError(str(e))
    results: List[ScoreOutWithContext] = []
    for chunk in reader:
        items = _rows_to_booking(chunk.to_dict(orient="records"))
        results.extend(score_items(items))
    return results
