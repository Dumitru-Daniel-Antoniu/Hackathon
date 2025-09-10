# Home.py
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import UploadFile, File
import io
from services.feature_engineering import add_engineered_features
from routes.scoring import router as scoring_router


app = FastAPI(title = "Booking Risk Scorer", version="1.0")
app.include_router(scoring_router)

@app.get("/health")
def health():
    return {"status": "ok"}
