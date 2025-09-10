from pydantic import BaseModel, Field, AliasChoices
from typing import Optional

class BookingIn(BaseModel):
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

class SimpleBooking(BaseModel):
    typical_horizon: float = Field(..., ge=0,
        validation_alias=AliasChoices("typical_horizon", "TYPICAL_HORIZON")
    )
    base_fdr: float = Field(..., ge=0,
        validation_alias=AliasChoices("base_fdr", "BASE_FDR")
    )
    shock_flag: int = Field(..., ge=0, le=1,
        validation_alias=AliasChoices("shock_flag", "SHOCK_FLAG")
    )
    days_in_advance: float = Field(..., ge=0,
        validation_alias=AliasChoices("days_in_advance", "DAYS_IN_ADVANCE")
    )
    booking_amount: float = Field(..., ge=0,
        validation_alias=AliasChoices("booking_amount", "BOOKING_AMOUNT")
    )
    vertical: Optional[str] = Field( None,
        validation_alias=AliasChoices("vertical", "VERTICAL")
    )
    country: Optional[str] = Field( None,
        validation_alias=AliasChoices("country", "COUNTRY")
    )