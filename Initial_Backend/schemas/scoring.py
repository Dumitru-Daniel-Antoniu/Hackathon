from typing import Literal, Optional

from pydantic import BaseModel


class ScoreOut(BaseModel):
    probability: float
    risk_score: float
    risk_tier: Literal["Trusted Partner","Established Operator","Developing Organization","High-Risk Counterparty","Fraudulent Actor"]
    suggested_reserve_percent: float
    suggested_settlement_delay_days: int

class ScoreOutWithContext(ScoreOut):
    merchant_id: Optional[str] = None
    vertical: Optional[str] = None
    country: Optional[str] = None
    days_in_advance: Optional[float] = None
    booking_amount: Optional[float] = None
