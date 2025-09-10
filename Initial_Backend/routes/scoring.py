from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

from schemas.bookings import Booking, SimpleBooking
from schemas.scoring import ScoreOutWithContext
from services.risk_scoring_service import score_items, score_csv_bytes, RowValidationError, CsvParseError, \
    SIMPLE_DEFAULTS

router = APIRouter()

@router.post("/score/individual")
async def score(payload: Booking):
    try:
        return score_items([payload.model_dump()])[0]
    except RowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.post("/score/individual/simple", response_model=ScoreOutWithContext)
async def score_individual_simple(payload: SimpleBooking) -> ScoreOutWithContext:
    try:

        row = {**SIMPLE_DEFAULTS, **payload.model_dump(exclude_none=True)}
        result = score_items([row])[0]
        return ScoreOutWithContext(**result)
    except RowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@router.post("/score/batch")
async def score_batch(payloads: List[Booking]):
    try:
        return score_items([p.model_dump() for p in payloads])
    except RowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.post("/score/csv", response_model=List[ScoreOutWithContext])
async def score_csv(file: UploadFile = File(...)) -> List[ScoreOutWithContext]:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    data = await file.read()
    try:
        rows = score_csv_bytes(data)  # -> list[dict]
        return [ScoreOutWithContext(**r) for r in rows]
    except CsvParseError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
    except RowValidationError as e:
        raise HTTPException(status_code=400, detail=f"Row validation error: {e}")