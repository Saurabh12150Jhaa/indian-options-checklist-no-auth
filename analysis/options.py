"""NSE option-chain parsing, OI analytics, IV analysis, and max-pain calculation."""

from typing import Optional

import numpy as np
import pandas as pd

from core.models import BULLISH, BEARISH, NEUTRAL


# ══════════════════════════════════════════════════════════════════════════
#  NSE OPTION CHAIN PARSING & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════


def parse_nse_option_chain(nse_data: dict, expiry: Optional[str] = None) -> pd.DataFrame:
    """
    Parse NSE option-chain JSON into a flat DataFrame.
    If *expiry* is given, filter to that expiry; otherwise use all (filtered totals).
    """
    records_key = "filtered" if expiry is None else "records"
    data_list = nse_data.get(records_key, {}).get("data", [])

    rows = []
    for item in data_list:
        if expiry and item.get("expiryDate") != expiry:
            continue

        strike = item.get("strikePrice", 0)
        ce = item.get("CE", {})
        pe = item.get("PE", {})

        rows.append({
            "strike": float(strike),
            "call_oi": ce.get("openInterest", 0),
            "call_chg_oi": ce.get("changeinOpenInterest", 0),
            "call_volume": ce.get("totalTradedVolume", 0),
            "call_iv": ce.get("impliedVolatility", 0),
            "call_ltp": ce.get("lastPrice", 0),
            "call_bid": ce.get("bidprice", 0),
            "call_ask": ce.get("askPrice", 0),
            "put_oi": pe.get("openInterest", 0),
            "put_chg_oi": pe.get("changeinOpenInterest", 0),
            "put_volume": pe.get("totalTradedVolume", 0),
            "put_iv": pe.get("impliedVolatility", 0),
            "put_ltp": pe.get("lastPrice", 0),
            "put_bid": pe.get("bidprice", 0),
            "put_ask": pe.get("askPrice", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("strike", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def get_expiry_dates(nse_data: dict) -> list[str]:
    """Extract sorted list of available expiry dates from NSE payload."""
    return nse_data.get("records", {}).get("expiryDates", [])


def get_underlying_value(nse_data: dict) -> float:
    """Extract the underlying spot price from NSE payload."""
    return float(nse_data.get("records", {}).get("underlyingValue", 0))


def compute_pcr(chain_df: pd.DataFrame) -> Optional[float]:
    if chain_df.empty:
        return None
    total_put = chain_df["put_oi"].sum()
    total_call = chain_df["call_oi"].sum()
    if total_call == 0:
        return None
    return round(total_put / total_call, 3)


def compute_pcr_volume(chain_df: pd.DataFrame) -> Optional[float]:
    if chain_df.empty:
        return None
    total_put = chain_df["put_volume"].sum()
    total_call = chain_df["call_volume"].sum()
    if total_call == 0:
        return None
    return round(total_put / total_call, 3)


def compute_max_pain(chain_df: pd.DataFrame) -> Optional[float]:
    """Max Pain = strike where total option-buyer payout is minimised."""
    if chain_df.empty:
        return None

    strikes = chain_df["strike"].values
    call_oi = chain_df["call_oi"].values
    put_oi = chain_df["put_oi"].values

    min_pain = float("inf")
    max_pain_strike = strikes[0]

    for i, test in enumerate(strikes):
        call_payout = np.sum(np.maximum(test - strikes, 0) * call_oi)
        put_payout = np.sum(np.maximum(strikes - test, 0) * put_oi)
        total = call_payout + put_payout
        if total < min_pain:
            min_pain = total
            max_pain_strike = test

    return round(float(max_pain_strike), 2)


def compute_highest_oi_strikes(chain_df: pd.DataFrame) -> dict:
    if chain_df.empty:
        return {"highest_call_oi_strike": None, "highest_put_oi_strike": None}
    ci = chain_df["call_oi"].idxmax()
    pi = chain_df["put_oi"].idxmax()
    return {
        "highest_call_oi_strike": float(chain_df.loc[ci, "strike"]),
        "highest_call_oi_value": int(chain_df.loc[ci, "call_oi"]),
        "highest_put_oi_strike": float(chain_df.loc[pi, "strike"]),
        "highest_put_oi_value": int(chain_df.loc[pi, "put_oi"]),
    }


def compute_oi_buildup(chain_df: pd.DataFrame, spot: float, band: int = 10) -> dict:
    """
    Analyse where fresh OI is being built up near the spot.
    Positive call OI change near spot = resistance building.
    Positive put OI change near spot = support building.
    """
    if chain_df.empty:
        return {"call_buildup": 0, "put_buildup": 0, "buildup_signal": NEUTRAL}

    chain_df = chain_df.copy()
    chain_df["_dist"] = (chain_df["strike"] - spot).abs()
    near = chain_df.nsmallest(band * 2, "_dist")

    call_build = int(near["call_chg_oi"].sum())
    put_build = int(near["put_chg_oi"].sum())

    if put_build > call_build * 1.3:
        sig = BULLISH  # Puts being written -> support
    elif call_build > put_build * 1.3:
        sig = BEARISH  # Calls being written -> resistance
    else:
        sig = NEUTRAL

    return {"call_buildup": call_build, "put_buildup": put_build, "buildup_signal": sig}


def compute_iv_summary(chain_df: pd.DataFrame, spot: float) -> dict:
    if chain_df.empty:
        return {"atm_call_iv": 0, "atm_put_iv": 0, "iv_skew": 0}
    chain_df = chain_df.copy()
    chain_df["_dist"] = (chain_df["strike"] - spot).abs()
    atm = chain_df.nsmallest(5, "_dist")
    avg_call_iv = round(float(atm["call_iv"].mean()), 2)
    avg_put_iv = round(float(atm["put_iv"].mean()), 2)
    return {
        "atm_call_iv": avg_call_iv,
        "atm_put_iv": avg_put_iv,
        "iv_skew": round(avg_put_iv - avg_call_iv, 2),
    }


def compute_atm_straddle(chain_df: pd.DataFrame, spot: float) -> Optional[dict]:
    """
    ATM straddle price = call LTP + put LTP at the strike nearest to spot.
    This gives the market's expected move for the expiry.
    """
    if chain_df.empty:
        return None
    chain_df = chain_df.copy()
    chain_df["_dist"] = (chain_df["strike"] - spot).abs()
    atm_row = chain_df.loc[chain_df["_dist"].idxmin()]
    straddle = float(atm_row["call_ltp"]) + float(atm_row["put_ltp"])
    pct = (straddle / spot) * 100 if spot else 0
    return {
        "atm_strike": float(atm_row["strike"]),
        "straddle_price": round(straddle, 2),
        "expected_move_pct": round(pct, 2),
        "upper_range": round(spot + straddle, 2),
        "lower_range": round(spot - straddle, 2),
    }


def run_options_analysis(nse_data: dict, expiry: Optional[str] = None) -> dict:
    """Full options chain analysis from raw NSE payload."""
    spot = get_underlying_value(nse_data)
    chain_df = parse_nse_option_chain(nse_data, expiry)

    pcr = compute_pcr(chain_df)
    pcr_vol = compute_pcr_volume(chain_df)
    max_pain = compute_max_pain(chain_df)
    oi_strikes = compute_highest_oi_strikes(chain_df)
    oi_buildup = compute_oi_buildup(chain_df, spot)
    iv_summary = compute_iv_summary(chain_df, spot)
    atm_straddle = compute_atm_straddle(chain_df, spot)

    return {
        "spot": spot,
        "chain_df": chain_df,
        "pcr_oi": pcr,
        "pcr_volume": pcr_vol,
        "max_pain": max_pain,
        **oi_strikes,
        **oi_buildup,
        **iv_summary,
        "atm_straddle": atm_straddle,
    }
