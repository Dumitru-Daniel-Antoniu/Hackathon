from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

from schemas.bookings import Booking
from schemas.scoring import ScoreOutWithContext
from services.risk_scoring_service import score_items, score_csv_bytes, RowValidationError, CsvParseError

router = APIRouter()

@router.post("/score/individual")
async def score(payload: Booking):
    try:
        # ✅ convert to dict
        return score_items([payload.model_dump()])[0]
    except RowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.post("/score/batch")
async def score_batch(payloads: List[Booking]):
    try:
        # ✅ convert each to dict
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
        # ✅ enforce response schema
        return [ScoreOutWithContext(**r) for r in rows]
    except CsvParseError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
    except RowValidationError as e:
        raise HTTPException(status_code=400, detail=f"Row validation error: {e}")