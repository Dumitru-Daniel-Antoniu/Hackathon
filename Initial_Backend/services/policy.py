from typing import Tuple

# Tunable constants (easy to calibrate without touching code paths)
RESERVE_MIN = 2.5  # % held back even for very low risk
RESERVE_MAX = 50.0  # hard cap (business rule)


def _policy(prob: float, days: float) -> Tuple[float, int, str]:
    reserve_pct = 0
    pct = prob * 100
    delayed_days = min(45.0, 15.0 * prob + 0.08 * days)
    if 0 <= pct <= 10.5:
        tier = "Trusted Partner"
        reserve_pct = 5
    elif 10.5 < pct <= 15:
        tier = "Established Operator"
        reserve_pct = 10
    elif 15 < pct <= 35:
        tier = "Developing Organization"
        reserve_pct = 25
    elif 35 < pct <= 50:
        tier = "High-Risk Counterparty"
        reserve_pct = 50
    else:  # >50 (shouldn’t occur with RESERVE_MAX=50, but kept for completeness)
        tier = "Fraudulent Actor"
        reserve_pct = 75

    return reserve_pct, round(delayed_days), tier