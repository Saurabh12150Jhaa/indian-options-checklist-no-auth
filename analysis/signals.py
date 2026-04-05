"""Signal aggregation, pre-trade checklist generation, and market-phase detection."""

from datetime import datetime
from typing import Optional

import pandas as pd

from config import IST, MARKET_OPEN, MARKET_CLOSE, PRE_MARKET_OPEN
from core.models import BULLISH, BEARISH, NEUTRAL


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
