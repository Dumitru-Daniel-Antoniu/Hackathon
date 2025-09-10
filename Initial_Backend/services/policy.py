import numpy as np
from typing import Tuple

def _policy(prob: float, days: float):
    days_ahead = float(days) if days is not None else 0.0

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

    reserve_pct = float(np.clip(100*(0.08*prob + 0.0009*days_ahead), 0, 50))
    delay_days  = int(np.clip(15*prob + 0.08*days_ahead, 0, 45))

    return round(reserve_pct, 3), int(round(delay_days)), tier