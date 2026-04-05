"""
Analysis module – technical indicators, NSE option-chain analytics,
signal aggregation, and market-hours awareness.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    EMA_PERIODS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    IST, MARKET_OPEN, MARKET_CLOSE, PRE_MARKET_OPEN,
)

# ── Signal labels ──────────────────────────────────────────────────────────
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"


# ══════════════════════════════════════════════════════════════════════════
#  PURE-PANDAS TA HELPERS (no external TA library needed)
# ══════════════════════════════════════════════════════════════════════════


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average using pandas ewm."""
    if len(series) < length:
        return pd.Series(dtype=float)
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line, "signal": signal_line, "histogram": histogram,
    })


# ══════════════════════════════════════════════════════════════════════════
#  MARKET HOURS
# ══════════════════════════════════════════════════════════════════════════


def market_phase() -> str:
    """Return 'pre_market', 'market_open', or 'post_market'."""
    now = datetime.now(IST).time()
    if now < PRE_MARKET_OPEN:
        return "pre_market"
    if now < MARKET_OPEN:
        return "pre_market"
    if now <= MARKET_CLOSE:
        return "market_open"
    return "post_market"


# ══════════════════════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════


def compute_emas(df: pd.DataFrame) -> dict[int, float]:
    emas: dict[int, float] = {}
    close = df["close"]
    for p in EMA_PERIODS:
        series = _ema(close, length=p)
        if not series.empty:
            vals = series.dropna()
            if not vals.empty:
                emas[p] = round(float(vals.iloc[-1]), 2)
    return emas


def compute_rsi(df: pd.DataFrame) -> Optional[float]:
    series = _rsi(df["close"], length=RSI_PERIOD)
    if series is None:
        return None
    vals = series.dropna()
    return round(float(vals.iloc[-1]), 2) if not vals.empty else None


def compute_macd(df: pd.DataFrame) -> Optional[dict]:
    macd_df = _macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_df is None or macd_df.empty:
        return None
    latest = macd_df.dropna()
    if latest.empty:
        return None
    row = latest.iloc[-1]
    return {
        "macd": round(float(row["macd"]), 2),
        "signal": round(float(row["signal"]), 2),
        "histogram": round(float(row["histogram"]), 2),
    }


def compute_pivot_points(df: pd.DataFrame) -> Optional[dict]:
    """
    Classic pivot points from the *last completed* trading day.
    During market hours the last row is the current incomplete day,
    so we use iloc[-2]. After market close, the last row is the
    completed day itself — still use iloc[-2] to get the *previous*
    completed day, since pivots should be calculated from yesterday's
    data to define today's levels. Requires at least 2 rows.
    """
    if len(df) < 2:
        return None
    # Always use second-to-last row: during market hours this is yesterday;
    # after close this is also yesterday (last row = today's completed bar).
    prev = df.iloc[-2]
    h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
    if h == 0 or l == 0:
        return None
    pp = (h + l + c) / 3
    return {
        "PP": round(pp, 2),
        "R1": round(2 * pp - l, 2),
        "R2": round(pp + (h - l), 2),
        "R3": round(h + 2 * (pp - l), 2),
        "S1": round(2 * pp - h, 2),
        "S2": round(pp - (h - l), 2),
        "S3": round(l - 2 * (h - pp), 2),
    }


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> Optional[dict]:
    subset = df.tail(lookback)
    if subset.empty:
        return None
    swing_high = float(subset["high"].max())
    swing_low = float(subset["low"].min())
    diff = swing_high - swing_low
    if diff == 0:
        return None
    return {
        "Swing High": round(swing_high, 2),
        "0.236": round(swing_high - 0.236 * diff, 2),
        "0.382": round(swing_high - 0.382 * diff, 2),
        "0.500": round(swing_high - 0.5 * diff, 2),
        "0.618": round(swing_high - 0.618 * diff, 2),
        "0.786": round(swing_high - 0.786 * diff, 2),
        "Swing Low": round(swing_low, 2),
    }


def run_technical_analysis(df: pd.DataFrame) -> dict:
    spot = round(float(df["close"].iloc[-1]), 2) if not df.empty else 0
    return {
        "spot": spot,
        "emas": compute_emas(df),
        "rsi": compute_rsi(df),
        "macd": compute_macd(df),
        "pivots": compute_pivot_points(df),
        "fibonacci": compute_fibonacci_levels(df),
    }


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


# ══════════════════════════════════════════════════════════════════════════
#  SIGNAL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════


def _ema_signal(spot: float, emas: dict) -> str:
    if not emas:
        return NEUTRAL
    above = sum(1 for v in emas.values() if spot > v)
    ratio = above / len(emas)
    if ratio >= 0.75:
        return BULLISH
    if ratio <= 0.25:
        return BEARISH
    return NEUTRAL


def _rsi_signal(rsi: Optional[float]) -> str:
    if rsi is None:
        return NEUTRAL
    if rsi > 60:
        return BULLISH
    if rsi < 40:
        return BEARISH
    return NEUTRAL


def _macd_signal(macd_data: Optional[dict]) -> str:
    if macd_data is None:
        return NEUTRAL
    if macd_data["histogram"] > 0 and macd_data["macd"] > macd_data["signal"]:
        return BULLISH
    if macd_data["histogram"] < 0 and macd_data["macd"] < macd_data["signal"]:
        return BEARISH
    return NEUTRAL


def _pcr_signal(pcr: Optional[float]) -> str:
    if pcr is None:
        return NEUTRAL
    if pcr > 1.2:
        return BULLISH
    if pcr < 0.7:
        return BEARISH
    return NEUTRAL


def _vix_signal(vix: Optional[float]) -> str:
    if vix is None:
        return NEUTRAL
    if vix < 13:
        return BULLISH
    if vix > 20:
        return BEARISH
    return NEUTRAL


def _global_cues_signal(global_df: pd.DataFrame) -> str:
    if global_df.empty:
        return NEUTRAL
    avg = global_df["Change %"].mean()
    if avg > 0.3:
        return BULLISH
    if avg < -0.3:
        return BEARISH
    return NEUTRAL


def _max_pain_signal(spot: float, max_pain: Optional[float]) -> str:
    if max_pain is None or spot == 0:
        return NEUTRAL
    diff_pct = ((spot - max_pain) / max_pain) * 100
    if diff_pct > 0.5:
        return BEARISH
    if diff_pct < -0.5:
        return BULLISH
    return NEUTRAL


def _oi_levels_signal(spot: float, call_strike: Optional[float], put_strike: Optional[float]) -> str:
    if call_strike is None or put_strike is None:
        return NEUTRAL
    mid = (call_strike + put_strike) / 2
    if spot > mid:
        return BULLISH
    if spot < mid:
        return BEARISH
    return NEUTRAL


def generate_checklist(
    technicals: dict,
    options_data: dict,
    vix: Optional[float],
    global_df: pd.DataFrame,
    fii_dii_df: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """
    Build the 10-point pre-trade checklist.
    Each item: {indicator, value (str), signal, weight}
    All values are normalised to str for consistent rendering.
    """
    # Prefer NSE spot (real-time) over yfinance spot (delayed)
    nse_spot = options_data.get("spot", 0)
    yf_spot = technicals.get("spot", 0)
    spot = nse_spot if nse_spot else yf_spot
    items = []

    # 1. Global Cues
    g_sig = _global_cues_signal(global_df)
    if not global_df.empty:
        avg_chg = round(float(global_df["Change %"].mean()), 2)
        items.append({"indicator": "Global Cues", "value": f"Avg {avg_chg:+.2f}%", "signal": g_sig, "weight": 1.0})
    else:
        items.append({"indicator": "Global Cues", "value": "N/A", "signal": NEUTRAL, "weight": 1.0})

    # 2. India VIX
    vix_str = f"{vix:.2f}" if vix else "N/A"
    items.append({"indicator": "India VIX", "value": vix_str, "signal": _vix_signal(vix), "weight": 1.5})

    # 3. EMA Trend
    emas = technicals.get("emas", {})
    above = sum(1 for v in emas.values() if spot > v)
    items.append({"indicator": "EMA Trend", "value": f"{above}/{len(emas)} above", "signal": _ema_signal(spot, emas), "weight": 1.5})

    # 4. RSI
    rsi = technicals.get("rsi")
    rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
    items.append({"indicator": "RSI (14)", "value": rsi_str, "signal": _rsi_signal(rsi), "weight": 1.0})

    # 5. MACD
    macd = technicals.get("macd")
    hist_str = f"{macd['histogram']:+.2f}" if macd else "N/A"
    items.append({"indicator": "MACD", "value": f"Hist: {hist_str}", "signal": _macd_signal(macd), "weight": 1.0})

    # 6. PCR (OI)
    pcr = options_data.get("pcr_oi")
    pcr_str = f"{pcr:.3f}" if pcr is not None else "N/A"
    items.append({"indicator": "PCR (OI)", "value": pcr_str, "signal": _pcr_signal(pcr), "weight": 1.5})

    # 7. Max Pain
    mp = options_data.get("max_pain")
    mp_str = f"{mp:,.0f} (Spot: {spot:,.0f})" if mp else "N/A"
    items.append({"indicator": "Max Pain", "value": mp_str, "signal": _max_pain_signal(spot, mp), "weight": 1.0})

    # 8. OI Levels
    cs = options_data.get("highest_call_oi_strike")
    ps = options_data.get("highest_put_oi_strike")
    oi_val = f"S: {ps:,.0f} / R: {cs:,.0f}" if cs and ps else "N/A"
    items.append({"indicator": "OI Levels", "value": oi_val, "signal": _oi_levels_signal(spot, cs, ps), "weight": 1.5})

    # 9. OI Buildup
    buildup_sig = options_data.get("buildup_signal", NEUTRAL)
    cb = options_data.get("call_buildup", 0)
    pb = options_data.get("put_buildup", 0)
    items.append({"indicator": "OI Buildup", "value": f"Call: {cb:+,} / Put: {pb:+,}", "signal": buildup_sig, "weight": 1.0})

    # 10. FII Activity
    fii_sig = NEUTRAL
    fii_val = "N/A"
    if fii_dii_df is not None and not fii_dii_df.empty:
        fii_row = fii_dii_df[fii_dii_df["Category"].str.contains("FII|FPI", case=False, na=False)]
        if not fii_row.empty:
            net = fii_row["Net Value (Cr)"].iloc[0]
            fii_val = f"Net: {net} Cr"
            try:
                net_f = float(str(net).replace(",", ""))
                fii_sig = BULLISH if net_f > 0 else BEARISH
            except (ValueError, TypeError):
                pass
    items.append({"indicator": "FII Activity", "value": fii_val, "signal": fii_sig, "weight": 1.0})

    return items


def compute_overall_bias(checklist: list[dict]) -> tuple[str, float]:
    """
    Weighted scoring: BULLISH = +1, BEARISH = -1, NEUTRAL = 0.
    Returns (label, score_pct) where score_pct ranges from -100 to +100.
    """
    score = 0.0
    total_w = 0.0
    for item in checklist:
        w = item.get("weight", 1)
        total_w += w
        if item["signal"] == BULLISH:
            score += w
        elif item["signal"] == BEARISH:
            score -= w
    if total_w == 0:
        return NEUTRAL, 0.0
    pct = round((score / total_w) * 100, 1)
    if pct > 25:
        return BULLISH, pct
    if pct < -25:
        return BEARISH, pct
    return NEUTRAL, pct
