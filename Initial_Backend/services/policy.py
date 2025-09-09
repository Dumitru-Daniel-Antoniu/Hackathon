import numpy as np
from typing import Tuple

def post_policy(prob: float, days_ahead: float) -> Tuple[str, float, int]:
    """
    Map model probability -> tier, reserve %, settlement delay (days).
    Prob in [0,1], days_ahead can be None/NaN -> treat as 0.
    """
    days_ahead = float(days_ahead) if days_ahead is not None else 0.0

    if prob >= 0.41:
        tier = "Fraudulent Actor"
    elif prob >= 0.21:
        tier = "High-Risk Counterparty"
    elif prob >= 0.11:
        tier = "Developing Organization"
    elif prob >= 0.06:
        tier = "Established Operator"
    else:
        tier = "Trusted Partner"

    reserve_percent = float(np.clip(100*(0.08*prob + 0.0009*days_ahead), 0, 50))
    settlement_delay = int(np.clip(15*prob + 0.08*days_ahead, 0, 45))

    return tier, round(reserve_percent, 2), settlement_delay
