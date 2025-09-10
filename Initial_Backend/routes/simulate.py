# routes/simulate.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from services.policy import _policy  # uses your existing policy

router = APIRouter(prefix="", tags=["simulation"])

class SimulatePolicyIn(BaseModel):
    trust_score: float | None = Field(
        default=None, description="Merchant trust score (0–100). Higher = more trustworthy."
    )
    risk_score: float | None = Field(
        default=None, description="Model-like risk probability (0–1). Higher = riskier."
    )
    days_in_advance: float = Field(
        default=0.0, ge=0, description="Lead time between booking and delivery, in days."
    )

    @model_validator(mode="after")
    def _validate_inputs(self):
        if self.trust_score is None and self.risk_score is None:
            raise ValueError("Provide either trust_score (0–100) or risk_score (0–1).")
        if self.trust_score is not None and not (0.0 <= self.trust_score <= 100.0):
            raise ValueError("trust_score must be in [0, 100].")
        if self.risk_score is not None and not (0.0 <= self.risk_score <= 1.0):
            raise ValueError("risk_score must be in [0, 1].")
        return self

class SimulatePolicyOut(BaseModel):
    reserve_percent: float = Field(..., description="Percent of funds to retain in the bank.")
    # You can expose these for transparency, or remove if you truly only want the %:
    suggested_settlement_delay_days: int | None = None
    risk_tier: str | None = None
    effective_risk_score: float = Field(..., description="Risk used by the policy (0–1).")

@router.post("/simulate_policy", response_model=SimulatePolicyOut)
def simulate_policy(payload: SimulatePolicyIn):
    # Choose the risk signal
    prob = payload.risk_score
    if prob is None:
        # Derive a simple probability from trust (higher trust ⇒ lower risk)
        prob = 1.0 - (payload.trust_score / 100.0)

    # Run your existing policy (caps & tiering live there)
    reserve_pct, delay_days, tier = _policy(float(prob), float(payload.days_in_advance))

    return SimulatePolicyOut(
        reserve_percent=reserve_pct,
        suggested_settlement_delay_days=delay_days,
        risk_tier=tier,
        effective_risk_score=prob,
    )
