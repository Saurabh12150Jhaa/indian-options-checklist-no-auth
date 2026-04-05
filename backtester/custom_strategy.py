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


# ══════════════════════════════════════════════════════════════════════════
#  CONDITION REGISTRY
# ══════════════════════════════════════════════════════════════════════════

CONDITION_REGISTRY = {
    "rsi": {
        "eval": _eval_rsi,
        "needs": "ohlc",
        "label": "RSI",
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
        "params": [
            {"name": "period", "type": "int", "default": 20, "min": 5, "max": 200, "label": "EMA Period"},
            {"name": "position", "type": "select", "options": ["above", "below"], "default": "above", "label": "Position"},
        ],
    },
    "candle_pattern": {
        "eval": _eval_candle_pattern,
        "needs": "ohlc",
        "label": "Candle Pattern",
        "params": [
            {"name": "pattern", "type": "select", "options": ["engulfing", "pin_bar", "inside_bar"], "default": "engulfing", "label": "Pattern"},
            {"name": "direction", "type": "select", "options": ["BULLISH", "BEARISH"], "default": "BULLISH", "label": "Direction"},
        ],
    },
    "pcr": {
        "eval": _eval_pcr,
        "needs": "chain",
        "label": "Put-Call Ratio",
        "params": [
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "label": "PCR Value"},
        ],
    },
    "iv_rank": {
        "eval": _eval_iv_rank,
        "needs": "chain_spot",
        "label": "ATM IV Level",
        "params": [
            {"name": "operator", "type": "select", "options": [">", ">=", "<", "<="], "default": ">", "label": "Operator"},
            {"name": "value", "type": "float", "default": 15, "min": 1, "max": 100, "label": "IV (%)"},
        ],
    },
    "day_of_week": {
        "eval": _eval_day_of_week,
        "needs": "date",
        "label": "Day of Week",
        "params": [
            {"name": "days", "type": "multiselect", "options": ["mon", "tue", "wed", "thu", "fri"], "default": ["mon", "tue", "wed", "thu", "fri"], "label": "Trading Days"},
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
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "Custom Strategy")

    def _get_ohlc(self, engine, dt: date, lookback: int = 50) -> Optional[pd.DataFrame]:
        dates = [d for d in engine.trading_dates if d <= dt][-lookback:]
        if len(dates) < 10:
            return None
        rows = []
        for d in dates:
            price = engine.get_underlying_price(d)
            if price:
                rows.append({
                    "date": d,
                    "open": price * 0.998,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price,
                })
        return pd.DataFrame(rows) if len(rows) >= 10 else None

    def evaluate_conditions(self, engine, dt, chain, spot) -> bool:
        """Evaluate all entry conditions."""
        conditions = self.config.get("conditions", [])
        if not conditions:
            return True

        logic = self.config.get("condition_logic", "AND")
        ohlc = None
        results = []

        for cond in conditions:
            ctype = cond.get("type")
            params = cond.get("params", {})
            registry_entry = CONDITION_REGISTRY.get(ctype)
            if not registry_entry:
                logger.warning("Unknown condition type: %s", ctype)
                results.append(False)
                continue

            needs = registry_entry["needs"]
            if needs == "ohlc":
                if ohlc is None:
                    ohlc = self._get_ohlc(engine, dt)
                if ohlc is None:
                    results.append(False)
                    continue
                results.append(registry_entry["eval"](ohlc, params))
            elif needs == "chain":
                results.append(registry_entry["eval"](chain, params))
            elif needs == "chain_spot":
                results.append(registry_entry["eval"](chain, spot, params))
            elif needs == "date":
                results.append(registry_entry["eval"](dt, params))
            else:
                results.append(False)

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
}
