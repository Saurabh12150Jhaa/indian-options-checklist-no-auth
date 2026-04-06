"""
Custom strategy builder for user-defined options strategies.

Users define strategies declaratively via a dict/JSON config:
  - Entry conditions: technical + options + price action rules
  - Option legs: type, strike selection, action
  - Exit rules: DTE, stop-loss, target, time-based

The CustomStrategy class compiles these rules into a callable
that works with the BacktestEngine.
"""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from backtester.utils import get_recent_ohlc

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  CONDITION EVALUATORS
# ══════════════════════════════════════════════════════════════════════════


def _eval_rsi(ohlc: pd.DataFrame, params: dict) -> bool:
    """Check RSI condition. params: {period, operator, value}"""
    period = params.get("period", 14)
    if len(ohlc) < period + 1:
        return False
    delta = ohlc["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val):
        return False
    return _compare(val, params.get("operator", ">"), params.get("value", 50))


def _eval_ema_cross(ohlc: pd.DataFrame, params: dict) -> bool:
    """Check if fast EMA crossed above/below slow EMA. params: {fast, slow, direction}"""
    fast_p = params.get("fast", 9)
    slow_p = params.get("slow", 21)
    if len(ohlc) < slow_p + 2:
        return False
    fast = ohlc["close"].ewm(span=fast_p, adjust=False).mean()
    slow = ohlc["close"].ewm(span=slow_p, adjust=False).mean()
    direction = params.get("direction", "above")
    if direction == "above":
        return fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
    else:
        return fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]


def _eval_price_vs_ema(ohlc: pd.DataFrame, params: dict) -> bool:
    """Check price position relative to EMA. params: {period, position}"""
    period = params.get("period", 20)
    if len(ohlc) < period + 1:
        return False
    ema = ohlc["close"].ewm(span=period, adjust=False).mean()
    position = params.get("position", "above")
    if position == "above":
        return ohlc["close"].iloc[-1] > ema.iloc[-1]
    return ohlc["close"].iloc[-1] < ema.iloc[-1]


def _eval_candle_pattern(ohlc: pd.DataFrame, params: dict) -> bool:
    """Check for specific candlestick pattern. params: {pattern, direction}"""
    pattern = params.get("pattern", "engulfing")
    direction = params.get("direction", "BULLISH")

    if len(ohlc) < 3:
        return False

    curr = ohlc.iloc[-1]
    prev = ohlc.iloc[-2]

    if pattern == "engulfing":
        prev_body = prev["close"] - prev["open"]
        curr_body = curr["close"] - curr["open"]
        if direction == "BULLISH":
            return (prev_body < 0 and curr_body > 0
                    and curr["open"] <= prev["close"]
                    and curr["close"] >= prev["open"])
        else:
            return (prev_body > 0 and curr_body < 0
                    and curr["open"] >= prev["close"]
                    and curr["close"] <= prev["open"])

    elif pattern == "pin_bar":
        o, h, l, c = curr["open"], curr["high"], curr["low"], curr["close"]
        body = abs(c - o)
        if body == 0:
            return False
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        if direction == "BULLISH":
            return lower_wick > 2.5 * body and upper_wick < body
        else:
            return upper_wick > 2.5 * body and lower_wick < body

    elif pattern == "inside_bar":
        mother = ohlc.iloc[-3] if len(ohlc) >= 3 else None
        if mother is None:
            return False
        inside = prev
        if inside["high"] < mother["high"] and inside["low"] > mother["low"]:
            if direction == "BULLISH":
                return curr["close"] > mother["high"]
            else:
                return curr["close"] < mother["low"]

    return False


def _eval_pcr(chain: pd.DataFrame, params: dict) -> bool:
    """Check PCR condition. params: {operator, value}"""
    if chain.empty:
        return False
    total_put = chain["put_oi"].sum() if "put_oi" in chain.columns else 0
    total_call = chain["call_oi"].sum() if "call_oi" in chain.columns else 0
    if total_call == 0:
        return False
    pcr = total_put / total_call
    return _compare(pcr, params.get("operator", ">"), params.get("value", 1.0))


def _eval_iv_rank(chain: pd.DataFrame, spot: float, params: dict) -> bool:
    """Check if ATM IV is above/below threshold. params: {operator, value}"""
    if chain.empty or "call_iv" not in chain.columns:
        return False
    chain = chain.copy()
    chain["_dist"] = (chain["strike"] - spot).abs()
    atm = chain.nsmallest(3, "_dist")
    avg_iv = atm["call_iv"].mean()
    if pd.isna(avg_iv):
        return False
    return _compare(avg_iv, params.get("operator", ">"), params.get("value", 15))


def _eval_day_of_week(dt: date, params: dict) -> bool:
    """Check if current day matches. params: {days: ["mon", "thu"]}"""
    day_names = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri"}
    current = day_names.get(dt.weekday(), "")
    allowed = params.get("days", ["mon", "tue", "wed", "thu", "fri"])
    return current in allowed


def _compare(val: float, operator: str, threshold: float) -> bool:
    """Generic comparison."""
    if operator == ">":
        return val > threshold
    elif operator == ">=":
        return val >= threshold
    elif operator == "<":
        return val < threshold
    elif operator == "<=":
        return val <= threshold
    elif operator == "==":
        return abs(val - threshold) < 0.001
    return False


def _eval_supertrend(ohlc: pd.DataFrame, params: dict) -> bool:
    """Supertrend indicator for trend detection. params: {period, multiplier, signal}"""
    period = params.get("period", 10)
    multiplier = params.get("multiplier", 3.0)
    signal = params.get("signal", "bullish")
    if len(ohlc) < period + 2:
        return False
    hl = ohlc["high"] - ohlc["low"]
    hc = (ohlc["high"] - ohlc["close"].shift(1)).abs()
    lc = (ohlc["low"] - ohlc["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    mid = (ohlc["high"] + ohlc["low"]) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    # Walk through to determine trend
    st_dir = [True] * len(ohlc)  # True = bullish
    for i in range(period + 1, len(ohlc)):
        if ohlc["close"].iloc[i] > upper_band.iloc[i - 1]:
            st_dir[i] = True
        elif ohlc["close"].iloc[i] < lower_band.iloc[i - 1]:
            st_dir[i] = False
        else:
            st_dir[i] = st_dir[i - 1]
    is_bullish = st_dir[-1]
    if signal == "bullish":
        return is_bullish
    return not is_bullish


def _eval_bollinger(ohlc: pd.DataFrame, params: dict) -> bool:
    """Bollinger Bands position check. params: {period, num_std, position}"""
    period = params.get("period", 20)
    num_std = params.get("num_std", 2.0)
    position = params.get("position", "above_upper")
    if len(ohlc) < period + 1:
        return False
    sma = ohlc["close"].rolling(period).mean()
    std = ohlc["close"].rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    close = ohlc["close"].iloc[-1]
    bw = ((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1]) * 100 if sma.iloc[-1] else 0
    if position == "above_upper":
        return close > upper.iloc[-1]
    elif position == "below_lower":
        return close < lower.iloc[-1]
    elif position == "within":
        return lower.iloc[-1] <= close <= upper.iloc[-1]
    elif position == "squeeze":
        # Bandwidth below 4% indicates squeeze
        return bw < 4.0
    return False


def _eval_vwap(ohlc: pd.DataFrame, params: dict) -> bool:
    """Approximate VWAP from OHLC. params: {position}"""
    position = params.get("position", "above")
    if len(ohlc) < 5:
        return False
    typical = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
    # Approximate volume as range * base factor
    approx_vol = (ohlc["high"] - ohlc["low"]).replace(0, 1)
    vwap = (typical * approx_vol).cumsum() / approx_vol.cumsum()
    close = ohlc["close"].iloc[-1]
    if position == "above":
        return close > vwap.iloc[-1]
    return close < vwap.iloc[-1]


def _eval_oi_change(chain: pd.DataFrame, params: dict) -> bool:
    """OI buildup analysis. params: {buildup}"""
    if chain.empty:
        return False
    buildup = params.get("buildup", "put_writing")
    call_chg = chain["call_chg_oi"].sum() if "call_chg_oi" in chain.columns else 0
    put_chg = chain["put_chg_oi"].sum() if "put_chg_oi" in chain.columns else 0
    if buildup == "call_writing":
        return call_chg > 0 and call_chg > put_chg
    elif buildup == "put_writing":
        return put_chg > 0 and put_chg > call_chg
    elif buildup == "call_unwinding":
        return call_chg < 0
    elif buildup == "put_unwinding":
        return put_chg < 0
    elif buildup == "long_buildup":
        return put_chg > 0 and call_chg < 0  # Put writers + call unwinding = bullish
    elif buildup == "short_buildup":
        return call_chg > 0 and put_chg < 0  # Call writers + put unwinding = bearish
    return False


def _eval_gap(ohlc: pd.DataFrame, params: dict) -> bool:
    """Gap up/down detection. params: {direction, min_pct}"""
    direction = params.get("direction", "up")
    min_pct = params.get("min_pct", 0.3)
    if len(ohlc) < 2:
        return False
    prev_close = ohlc["close"].iloc[-2]
    curr_open = ohlc["open"].iloc[-1]
    if prev_close == 0:
        return False
    gap_pct = ((curr_open - prev_close) / prev_close) * 100
    if direction == "up":
        return gap_pct >= min_pct
    return gap_pct <= -min_pct


def _eval_consecutive_candles(ohlc: pd.DataFrame, params: dict) -> bool:
    """N consecutive green/red candles. params: {color, count}"""
    color = params.get("color", "green")
    count = params.get("count", 3)
    if len(ohlc) < count:
        return False
    recent = ohlc.tail(count)
    if color == "green":
        return all(recent["close"] > recent["open"])
    return all(recent["close"] < recent["open"])


def _eval_atr_rank(ohlc: pd.DataFrame, params: dict) -> bool:
    """ATR as % of price (volatility filter). params: {period, operator, value}"""
    period = params.get("period", 14)
    if len(ohlc) < period + 1:
        return False
    hl = ohlc["high"] - ohlc["low"]
    hc = (ohlc["high"] - ohlc["close"].shift(1)).abs()
    lc = (ohlc["low"] - ohlc["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    close = ohlc["close"].iloc[-1]
    if close == 0 or pd.isna(atr):
        return False
    atr_pct = (atr / close) * 100
    return _compare(atr_pct, params.get("operator", ">"), params.get("value", 1.5))


def _eval_trend_strength(ohlc: pd.DataFrame, params: dict) -> bool:
    """ADX-based trend strength. params: {period, operator, value}"""
    period = params.get("period", 14)
    if len(ohlc) < period * 2:
        return False
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    val = adx.iloc[-1]
    if pd.isna(val):
        return False
    return _compare(val, params.get("operator", ">"), params.get("value", 25))


# ══════════════════════════════════════════════════════════════════════════
#  CONDITION REGISTRY
# ══════════════════════════════════════════════════════════════════════════

CONDITION_REGISTRY = {
    "rsi": {
        "eval": _eval_rsi,
        "needs": "ohlc",
        "label": "RSI",
        "description": "RSI momentum oscillator",
        "params": [
            {"name": "period", "type": "int", "default": 14, "min": 5, "max": 50, "label": "Period"},
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 50, "min": 0, "max": 100, "label": "Value"},
        ],
    },
    "ema_cross": {
        "eval": _eval_ema_cross,
        "needs": "ohlc",
        "label": "EMA Crossover",
        "description": "Moving average crossover signal",
        "params": [
            {"name": "fast", "type": "int", "default": 9, "min": 3, "max": 50, "label": "Fast EMA"},
            {"name": "slow", "type": "int", "default": 21, "min": 10, "max": 200, "label": "Slow EMA"},
            {"name": "direction", "type": "select", "options": ["above", "below"], "default": "above", "label": "Cross Direction"},
        ],
    },
    "price_vs_ema": {
        "eval": _eval_price_vs_ema,
        "needs": "ohlc",
        "label": "Price vs EMA",
        "description": "Price position relative to moving average",
        "params": [
            {"name": "period", "type": "int", "default": 20, "min": 5, "max": 200, "label": "EMA Period"},
            {"name": "position", "type": "select", "options": ["above", "below"], "default": "above", "label": "Position"},
        ],
    },
    "candle_pattern": {
        "eval": _eval_candle_pattern,
        "needs": "ohlc",
        "label": "Candle Pattern",
        "description": "Japanese candlestick pattern recognition",
        "params": [
            {"name": "pattern", "type": "select", "options": ["engulfing", "pin_bar", "inside_bar"], "default": "engulfing", "label": "Pattern"},
            {"name": "direction", "type": "select", "options": ["BULLISH", "BEARISH"], "default": "BULLISH", "label": "Direction"},
        ],
    },
    "pcr": {
        "eval": _eval_pcr,
        "needs": "chain",
        "label": "Put-Call Ratio",
        "description": "Put-Call Ratio from options chain",
        "params": [
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "label": "PCR Value"},
        ],
    },
    "iv_rank": {
        "eval": _eval_iv_rank,
        "needs": "chain_spot",
        "label": "ATM IV Level",
        "description": "ATM implied volatility level",
        "params": [
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 15, "min": 1, "max": 100, "label": "IV (%)"},
        ],
    },
    "day_of_week": {
        "eval": _eval_day_of_week,
        "needs": "date",
        "label": "Day of Week",
        "description": "Filter by specific trading days",
        "params": [
            {"name": "days", "type": "multiselect", "options": ["mon", "tue", "wed", "thu", "fri"], "default": ["mon", "tue", "wed", "thu", "fri"], "label": "Trading Days"},
        ],
    },
    "supertrend": {
        "eval": _eval_supertrend,
        "needs": "ohlc",
        "label": "Supertrend",
        "description": "Supertrend indicator for trend detection",
        "params": [
            {"name": "period", "type": "int", "default": 10, "min": 5, "max": 50, "label": "ATR Period"},
            {"name": "multiplier", "type": "float", "default": 3.0, "min": 1.0, "max": 5.0, "label": "Multiplier"},
            {"name": "signal", "type": "select", "options": ["bullish", "bearish"], "default": "bullish", "label": "Signal"},
        ],
    },
    "bollinger": {
        "eval": _eval_bollinger,
        "needs": "ohlc",
        "label": "Bollinger Bands",
        "description": "Price position relative to Bollinger Bands",
        "params": [
            {"name": "period", "type": "int", "default": 20, "min": 10, "max": 50, "label": "Period"},
            {"name": "num_std", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "label": "Std Dev"},
            {"name": "position", "type": "select", "options": ["above_upper", "below_lower", "within", "squeeze"], "default": "above_upper", "label": "Price Position"},
        ],
    },
    "vwap": {
        "eval": _eval_vwap,
        "needs": "ohlc",
        "label": "VWAP Position",
        "description": "Price vs Volume-Weighted Average Price",
        "params": [
            {"name": "position", "type": "select", "options": ["above", "below"], "default": "above", "label": "Position"},
        ],
    },
    "oi_change": {
        "eval": _eval_oi_change,
        "needs": "chain",
        "label": "OI Buildup",
        "description": "Open Interest change pattern analysis",
        "params": [
            {"name": "buildup", "type": "select", "options": ["call_writing", "put_writing", "call_unwinding", "put_unwinding", "long_buildup", "short_buildup"], "default": "put_writing", "label": "Buildup Type"},
        ],
    },
    "gap": {
        "eval": _eval_gap,
        "needs": "ohlc",
        "label": "Gap Up/Down",
        "description": "Detect gap up or gap down opening",
        "params": [
            {"name": "direction", "type": "select", "options": ["up", "down"], "default": "up", "label": "Direction"},
            {"name": "min_pct", "type": "float", "default": 0.3, "min": 0.1, "max": 5.0, "label": "Min Gap %"},
        ],
    },
    "consecutive_candles": {
        "eval": _eval_consecutive_candles,
        "needs": "ohlc",
        "label": "Consecutive Candles",
        "description": "N consecutive green or red candles",
        "params": [
            {"name": "color", "type": "select", "options": ["green", "red"], "default": "green", "label": "Candle Color"},
            {"name": "count", "type": "int", "default": 3, "min": 2, "max": 7, "label": "Count"},
        ],
    },
    "atr_rank": {
        "eval": _eval_atr_rank,
        "needs": "ohlc",
        "label": "ATR Volatility",
        "description": "ATR as percentage of price (volatility filter)",
        "params": [
            {"name": "period", "type": "int", "default": 14, "min": 5, "max": 50, "label": "ATR Period"},
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 1.5, "min": 0.1, "max": 10.0, "label": "ATR %"},
        ],
    },
    "trend_strength": {
        "eval": _eval_trend_strength,
        "needs": "ohlc",
        "label": "Trend Strength (ADX)",
        "description": "ADX-based trend strength measurement",
        "params": [
            {"name": "period", "type": "int", "default": 14, "min": 5, "max": 50, "label": "ADX Period"},
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 25, "min": 5, "max": 80, "label": "ADX Value"},
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  LEG TEMPLATES
# ══════════════════════════════════════════════════════════════════════════

LEG_TEMPLATES = {
    "atm_call": {"option_type": "CE", "offset_pct": 0, "label": "ATM Call"},
    "atm_put": {"option_type": "PE", "offset_pct": 0, "label": "ATM Put"},
    "otm_call_1": {"option_type": "CE", "offset_pct": 2.0, "label": "OTM Call (+2%)"},
    "otm_call_2": {"option_type": "CE", "offset_pct": 4.0, "label": "OTM Call (+4%)"},
    "otm_put_1": {"option_type": "PE", "offset_pct": -2.0, "label": "OTM Put (-2%)"},
    "otm_put_2": {"option_type": "PE", "offset_pct": -4.0, "label": "OTM Put (-4%)"},
    "itm_call": {"option_type": "CE", "offset_pct": -2.0, "label": "ITM Call (-2%)"},
    "itm_put": {"option_type": "PE", "offset_pct": 2.0, "label": "ITM Put (+2%)"},
    "otm_call_3": {"option_type": "CE", "offset_pct": 6.0, "label": "Deep OTM Call (+6%)"},
    "otm_put_3": {"option_type": "PE", "offset_pct": -6.0, "label": "Deep OTM Put (-6%)"},
    "otm_call_half": {"option_type": "CE", "offset_pct": 1.0, "label": "Slight OTM Call (+1%)"},
    "otm_put_half": {"option_type": "PE", "offset_pct": -1.0, "label": "Slight OTM Put (-1%)"},
    "otm_call_3pct": {"option_type": "CE", "offset_pct": 3.0, "label": "OTM Call (+3%)"},
    "otm_put_3pct": {"option_type": "PE", "offset_pct": -3.0, "label": "OTM Put (-3%)"},
    "deep_itm_call": {"option_type": "CE", "offset_pct": -4.0, "label": "Deep ITM Call (-4%)"},
    "deep_itm_put": {"option_type": "PE", "offset_pct": 4.0, "label": "Deep ITM Put (+4%)"},
}


# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM STRATEGY CLASS
# ══════════════════════════════════════════════════════════════════════════


class CustomStrategy:
    """
    Builds a strategy function from a declarative configuration.
    
    Config format:
    {
        "name": "My Strategy",
        "conditions": [
            {"type": "rsi", "params": {"period": 14, "operator": ">", "value": 60}},
            {"type": "price_vs_ema", "params": {"period": 20, "position": "above"}},
        ],
        "condition_logic": "AND",  # "AND" or "OR"
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
        ],
    }
    
    Supports nested condition groups:
    {
        "condition_groups": [
            {"conditions": [...], "logic": "AND"},
            {"conditions": [...], "logic": "AND"},
        ],
        "group_logic": "OR",
    }
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "Custom Strategy")

    def _eval_single_condition(self, cond, engine, dt, chain, spot, ohlc):
        """Evaluate a single condition, returning (result, ohlc) tuple."""
        ctype = cond.get("type")
        params = cond.get("params", {})
        registry_entry = CONDITION_REGISTRY.get(ctype)
        if not registry_entry:
            logger.warning("Unknown condition type: %s", ctype)
            return False, ohlc
        needs = registry_entry["needs"]
        if needs == "ohlc":
            if ohlc is None:
                ohlc = get_recent_ohlc(engine, dt)
            if ohlc is None:
                return False, ohlc
            return registry_entry["eval"](ohlc, params), ohlc
        elif needs == "chain":
            return registry_entry["eval"](chain, params), ohlc
        elif needs == "chain_spot":
            return registry_entry["eval"](chain, spot, params), ohlc
        elif needs == "date":
            return registry_entry["eval"](dt, params), ohlc
        return False, ohlc

    def evaluate_conditions(self, engine, dt, chain, spot) -> bool:
        """Evaluate all entry conditions."""
        ohlc = None

        # Support condition groups (nested logic)
        groups = self.config.get("condition_groups")
        if groups:
            group_logic = self.config.get("group_logic", "AND")
            group_results = []
            for group in groups:
                sub_conditions = group.get("conditions", [])
                sub_logic = group.get("logic", "AND")
                sub_results = []
                for cond in sub_conditions:
                    result, ohlc = self._eval_single_condition(cond, engine, dt, chain, spot, ohlc)
                    sub_results.append(result)
                if sub_logic == "AND":
                    group_results.append(all(sub_results) if sub_results else False)
                else:
                    group_results.append(any(sub_results) if sub_results else False)
            if group_logic == "AND":
                return all(group_results) if group_results else False
            return any(group_results) if group_results else False

        # Flat condition logic (original behavior)
        conditions = self.config.get("conditions", [])
        if not conditions:
            return True

        logic = self.config.get("condition_logic", "AND")
        results = []

        for cond in conditions:
            result, ohlc = self._eval_single_condition(cond, engine, dt, chain, spot, ohlc)
            results.append(result)

        if logic == "AND":
            return all(results)
        else:
            return any(results)

    def build_legs(self, engine, chain, spot, expiry) -> Optional[list[dict]]:
        """Build option legs from the leg configuration."""
        legs_config = self.config.get("legs", [])
        if not legs_config:
            return None

        result = []
        for leg_cfg in legs_config:
            template_name = leg_cfg.get("template")
            template = LEG_TEMPLATES.get(template_name)
            if not template:
                # Direct specification
                option_type = leg_cfg.get("option_type", "CE")
                offset = leg_cfg.get("offset_pct", 0)
            else:
                option_type = template["option_type"]
                offset = template["offset_pct"]

            # Allow override
            offset = leg_cfg.get("offset_pct", offset)

            # Support absolute points offset (converted to percentage)
            if "offset_points" in leg_cfg and spot > 0:
                offset = (leg_cfg["offset_points"] / spot) * 100

            strike = engine.find_strike_near_spot(chain, spot, option_type, offset_pct=offset)
            if strike is None:
                return None

            result.append({
                "option_type": option_type,
                "strike": strike,
                "action": leg_cfg.get("action", "BUY"),
                "qty": leg_cfg.get("qty", 1),
                "strategy_name": self.name,
            })

        return result if result else None

    def __call__(self, engine, dt, chain, spot, expiry) -> Optional[list[dict]]:
        """Strategy callable for the BacktestEngine."""
        if not self.evaluate_conditions(engine, dt, chain, spot):
            return None
        return self.build_legs(engine, chain, spot, expiry)


# ══════════════════════════════════════════════════════════════════════════
#  PRESET CUSTOM STRATEGIES
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_PRESETS = {
    "RSI Oversold Bull Spread": {
        "name": "RSI Oversold Bull Spread",
        "conditions": [
            {"type": "rsi", "params": {"period": 14, "operator": "<", "value": 35}},
            {"type": "price_vs_ema", "params": {"period": 200, "position": "above"}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
        ],
    },
    "High IV Iron Condor": {
        "name": "High IV Iron Condor",
        "conditions": [
            {"type": "iv_rank", "params": {"operator": ">", "value": 20}},
            {"type": "day_of_week", "params": {"days": ["mon", "tue"]}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
            {"template": "otm_call_2", "action": "BUY", "qty": 1},
            {"template": "otm_put_1", "action": "SELL", "qty": 1},
            {"template": "otm_put_2", "action": "BUY", "qty": 1},
        ],
    },
    "EMA Cross Momentum": {
        "name": "EMA Cross Momentum",
        "conditions": [
            {"type": "ema_cross", "params": {"fast": 9, "slow": 21, "direction": "above"}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
        ],
    },
    "Engulfing + High PCR": {
        "name": "Engulfing + High PCR",
        "conditions": [
            {"type": "candle_pattern", "params": {"pattern": "engulfing", "direction": "BULLISH"}},
            {"type": "pcr", "params": {"operator": ">", "value": 1.2}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
        ],
    },
    "Thursday Expiry Straddle Sell": {
        "name": "Thursday Expiry Straddle Sell",
        "conditions": [
            {"type": "day_of_week", "params": {"days": ["thu"]}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "SELL", "qty": 1},
            {"template": "atm_put", "action": "SELL", "qty": 1},
        ],
    },
    "Supertrend Bullish Call Buy": {
        "name": "Supertrend Bullish Call Buy",
        "conditions": [
            {"type": "supertrend", "params": {"period": 10, "multiplier": 3.0, "signal": "bullish"}},
            {"type": "trend_strength", "params": {"period": 14, "operator": ">", "value": 25}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
        ],
    },
    "Bollinger Squeeze Iron Fly": {
        "name": "Bollinger Squeeze Iron Fly",
        "conditions": [
            {"type": "bollinger", "params": {"period": 20, "num_std": 2.0, "position": "squeeze"}},
            {"type": "atr_rank", "params": {"period": 14, "operator": "<", "value": 1.0}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "SELL", "qty": 1},
            {"template": "atm_put", "action": "SELL", "qty": 1},
            {"template": "otm_call_1", "action": "BUY", "qty": 1},
            {"template": "otm_put_1", "action": "BUY", "qty": 1},
        ],
    },
    "Gap Down Recovery Bull Spread": {
        "name": "Gap Down Recovery Bull Spread",
        "conditions": [
            {"type": "gap", "params": {"direction": "down", "min_pct": 0.5}},
            {"type": "pcr", "params": {"operator": ">", "value": 1.3}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
        ],
    },
    "OI Short Buildup Bear Spread": {
        "name": "OI Short Buildup Bear Spread",
        "conditions": [
            {"type": "oi_change", "params": {"buildup": "short_buildup"}},
            {"type": "price_vs_ema", "params": {"period": 20, "position": "below"}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "atm_put", "action": "BUY", "qty": 1},
            {"template": "otm_put_1", "action": "SELL", "qty": 1},
        ],
    },
    "VWAP + 3 Green Candles Momentum": {
        "name": "VWAP + 3 Green Candles Momentum",
        "conditions": [
            {"type": "vwap", "params": {"position": "above"}},
            {"type": "consecutive_candles", "params": {"color": "green", "count": 3}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "otm_call_half", "action": "BUY", "qty": 1},
        ],
    },
    "Sideways Low ADX Strangle Sell": {
        "name": "Sideways Low ADX Strangle Sell",
        "conditions": [
            {"type": "trend_strength", "params": {"period": 14, "operator": "<", "value": 20}},
            {"type": "bollinger", "params": {"period": 20, "num_std": 2.0, "position": "within"}},
        ],
        "condition_logic": "AND",
        "legs": [
            {"template": "otm_call_1", "action": "SELL", "qty": 1},
            {"template": "otm_put_1", "action": "SELL", "qty": 1},
        ],
    },
    "Expiry Day Straddle with OI Confirmation": {
        "name": "Expiry Day Straddle with OI Confirmation",
        "condition_groups": [
            {
                "conditions": [
                    {"type": "day_of_week", "params": {"days": ["thu"]}},
                    {"type": "atr_rank", "params": {"period": 14, "operator": ">", "value": 1.2}},
                ],
                "logic": "AND",
            },
        ],
        "group_logic": "AND",
        "legs": [
            {"template": "atm_call", "action": "BUY", "qty": 1},
            {"template": "atm_put", "action": "BUY", "qty": 1},
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  STRATEGY DESCRIPTION HELPERS
# ══════════════════════════════════════════════════════════════════════════


def _describe_condition(ctype: str, params: dict) -> str:
    """Human-readable description of a single condition."""
    if ctype == "rsi":
        return f"RSI({params.get('period', 14)}) {params.get('operator', '>')} {params.get('value', 50)}"
    elif ctype == "ema_cross":
        d = params.get("direction", "above")
        return f"EMA {params.get('fast', 9)} crosses {d} EMA {params.get('slow', 21)}"
    elif ctype == "price_vs_ema":
        return f"Price {params.get('position', 'above')} EMA({params.get('period', 20)})"
    elif ctype == "candle_pattern":
        return f"{params.get('direction', 'BULLISH')} {params.get('pattern', 'engulfing')}"
    elif ctype == "pcr":
        return f"PCR {params.get('operator', '>')} {params.get('value', 1.0)}"
    elif ctype == "iv_rank":
        return f"ATM IV {params.get('operator', '>')} {params.get('value', 15)}%"
    elif ctype == "day_of_week":
        days = params.get("days", [])
        return f"Day is {', '.join(d.upper() for d in days)}"
    elif ctype == "supertrend":
        return f"Supertrend({params.get('period', 10)}, {params.get('multiplier', 3.0)}) is {params.get('signal', 'bullish')}"
    elif ctype == "bollinger":
        return f"Price {params.get('position', 'above_upper').replace('_', ' ')} Bollinger({params.get('period', 20)})"
    elif ctype == "vwap":
        return f"Price {params.get('position', 'above')} VWAP"
    elif ctype == "oi_change":
        return f"OI shows {params.get('buildup', 'put_writing').replace('_', ' ')}"
    elif ctype == "gap":
        return f"Gap {params.get('direction', 'up')} >= {params.get('min_pct', 0.3)}%"
    elif ctype == "consecutive_candles":
        return f"{params.get('count', 3)} consecutive {params.get('color', 'green')} candles"
    elif ctype == "atr_rank":
        return f"ATR% {params.get('operator', '>')} {params.get('value', 1.5)}%"
    elif ctype == "trend_strength":
        return f"ADX({params.get('period', 14)}) {params.get('operator', '>')} {params.get('value', 25)}"
    return ctype


def describe_strategy(config: dict) -> str:
    """Generate a plain-English description of a custom strategy config."""
    parts = []
    name = config.get("name", "Custom Strategy")
    parts.append(f"**{name}**")

    # Describe conditions
    conditions = config.get("conditions", [])
    groups = config.get("condition_groups", [])

    if conditions:
        logic = config.get("condition_logic", "AND")
        cond_strs = []
        for c in conditions:
            ctype = c.get("type", "")
            params = c.get("params", {})
            cond_strs.append(_describe_condition(ctype, params))
        joiner = " **AND** " if logic == "AND" else " **OR** "
        parts.append(f"Enter when: {joiner.join(cond_strs)}")
    elif groups:
        group_logic = config.get("group_logic", "AND")
        group_strs = []
        for g in groups:
            sub_conds = g.get("conditions", [])
            sub_logic = g.get("logic", "AND")
            sub_strs = [_describe_condition(c.get("type", ""), c.get("params", {})) for c in sub_conds]
            sub_joiner = " AND " if sub_logic == "AND" else " OR "
            group_strs.append(f"({sub_joiner.join(sub_strs)})")
        g_joiner = f" **{group_logic}** "
        parts.append(f"Enter when: {g_joiner.join(group_strs)}")
    else:
        parts.append("Enter: Every trading day (no conditions)")

    # Describe legs
    legs = config.get("legs", [])
    if legs:
        leg_strs = []
        for leg in legs:
            tmpl = leg.get("template", "")
            tmpl_info = LEG_TEMPLATES.get(tmpl, {})
            label = tmpl_info.get("label", tmpl)
            action = leg.get("action", "BUY")
            qty = leg.get("qty", 1)
            if "option_type" in leg and not tmpl:
                otype = leg["option_type"]
                offset = leg.get("offset_pct", 0)
                pts = leg.get("offset_points")
                if pts:
                    label = f"{otype} ATM{pts:+d}pts"
                elif offset:
                    label = f"{otype} ATM{offset:+.1f}%"
                else:
                    label = f"ATM {otype}"
            leg_strs.append(f"{action} {qty}x {label}")
        parts.append(f"Legs: {', '.join(leg_strs)}")

    return " | ".join(parts)
