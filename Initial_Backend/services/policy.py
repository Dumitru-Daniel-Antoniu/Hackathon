import numpy as np
from typing import Tuple

def _policy(prob: float, days: float):
    reserve_pct = min(50.0, 100.0*(0.08*prob + 0.0009*days))
    delay_days  = min(45.0, 15.0*prob + 0.08*days)
    tier = "low" if prob < 0.2 else ("medium" if prob < 0.5 else "high")

    return round(reserve_pct, 3), int(round(delay_days)), tier