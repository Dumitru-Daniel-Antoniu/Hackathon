import pandas as pd, requests, math, json

FEATURES = ["mcc","age_months","new_merchant","trust_score","prior_cb_rate","refund_rate",
            "cancel_rate","website_uptime","sentiment","sales_growth_3m",
            "payout_delay_days","reserve_percent","deposit_policy_percent",
            "days_in_advance","booking_amount","shock_flag"]

def score_via_api(df: pd.DataFrame, base_url: str, chunk=3000) -> pd.DataFrame:
    # ensure all features exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    df = df.reset_index(drop=True).copy()
    df["row_id"] = df.index.astype(str)

    results = []
    for i in range(0, len(df), chunk):
        sl = df.iloc[i:i+chunk]
        payload = {
            "items": [
                dict(row_id=str(int(idx)), **{f: float(sl.loc[idx, f]) for f in FEATURES})
                for idx in sl.index
            ]
        }
        r = requests.post(f"{base_url}/v1/score/batch", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        results += data["items"]

    # merge back
    res_df = pd.DataFrame(results)
    out = df.merge(res_df, on="row_id", how="left")
    return out