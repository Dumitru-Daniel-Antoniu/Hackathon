from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from schemas.bookings import BookingIn
from schemas.scoring import  ScoreOutWithContext
from services.risk_scoring_service import (
    score_items, score_csv_bytes, RowValidationError, CsvParseError
)

router = APIRouter(prefix="/score", tags=["scoring"])

@router.post(path = "/individual",response_model=ScoreOutWithContext)
async def score(payload: BookingIn) -> ScoreOutWithContext:
    return (await score_batch([payload]))[0]

@router.post("/batch", response_model=List[ScoreOutWithContext])
async def score_batch(payloads: List[BookingIn]) -> List[ScoreOutWithContext]:
    return score_items(payloads)

@router.post("/csv", response_model=List[ScoreOutWithContext])
async def score_csv(file: UploadFile = File(...)) -> List[ScoreOutWithContext]:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    data = await file.read()
    try:
        return score_csv_bytes(data)
    except CsvParseError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
    except RowValidationError as e:
        raise HTTPException(status_code=400, detail=f"Row validation error: {e}")
