from typing import Optional
from pydantic import BaseModel, Field


class BookingIn(BaseModel):
    # minimal set — add others if you collect them in your UI
    MERCHANT_ID: Optional[str] = None
    MCC: Optional[int] = None
    VERTICAL: Optional[str] = None
    COUNTRY: Optional[str] = None

    TRUST_SCORE: float
    PRIOR_CB_RATE: float
    REFUND_RATE: float
    CANCEL_RATE: float
    SENTIMENT: float
    SALES_GROWTH_3M: float

    PAYOUT_DELAY_DAYS: float
    RESERVE_PERCENT: float
    DEPOSIT_POLICY_PERCENT: float

    DAYS_IN_ADVANCE: float
    BOOKING_AMOUNT: float
    NEW_MERCHANT: int = Field(..., description="0 or 1")
    SHOCK_FLAG: int = Field(..., description="0 or 1")


class Booking(BaseModel):
    MERCHANT_ID: Optional[str] = None
    MCC: Optional[int] = None
    VERTICAL: str
    COUNTRY: str
    TRUST_SCORE: float
    PRIOR_CB_RATE: float
    REFUND_RATE: float
    CANCEL_RATE: float
    SENTIMENT: float
    SALES_GROWTH_3M: float
    PAYOUT_DELAY_DAYS: float
    RESERVE_PERCENT: float
    DEPOSIT_POLICY_PERCENT: float
    DAYS_IN_ADVANCE: float
    BOOKING_AMOUNT: float
    NEW_MERCHANT: int
    SHOCK_FLAG: int
