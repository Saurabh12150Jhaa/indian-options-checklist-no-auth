"""
Prebuilt options strategies for the backtester.

Each strategy function has the signature:
    fn(engine, dt, chain, spot, expiry) -> list[dict] | None

Each returned dict represents a leg:
    {option_type: 'CE'/'PE', strike: float, action: 'BUY'/'SELL',
     qty: int, strategy_name: str}
"""

from datetime import date
from typing import Optional

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════
#  SINGLE-LEG STRATEGIES
# ══════════════════════════════════════════════════════════════════════════


def short_straddle(engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date):
    """Sell ATM Call + ATM Put. Profit when market stays near spot."""
    ce_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
    pe_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
    if ce_strike is None or pe_strike is None:
        return None
    return [
        {"option_type": "CE", "strike": ce_strike, "action": "SELL", "qty": 1, "strategy_name": "Short Straddle"},
        {"option_type": "PE", "strike": pe_strike, "action": "SELL", "qty": 1, "strategy_name": "Short Straddle"},
    ]


def long_straddle(engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date):
    """Buy ATM Call + ATM Put. Profit on large moves in either direction."""
    ce_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
    pe_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
    if ce_strike is None or pe_strike is None:
        return None
    return [
        {"option_type": "CE", "strike": ce_strike, "action": "BUY", "qty": 1, "strategy_name": "Long Straddle"},
        {"option_type": "PE", "strike": pe_strike, "action": "BUY", "qty": 1, "strategy_name": "Long Straddle"},
    ]


def short_strangle(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    otm_pct: float = 2.0,
):
    """Sell OTM Call + OTM Put. Wider range than straddle."""
    ce_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct)
    pe_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-otm_pct)
    if ce_strike is None or pe_strike is None:
        return None
    return [
        {"option_type": "CE", "strike": ce_strike, "action": "SELL", "qty": 1, "strategy_name": "Short Strangle"},
        {"option_type": "PE", "strike": pe_strike, "action": "SELL", "qty": 1, "strategy_name": "Short Strangle"},
    ]


def long_strangle(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    otm_pct: float = 2.0,
):
    """Buy OTM Call + OTM Put. Cheaper than straddle, needs bigger move."""
    ce_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct)
    pe_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-otm_pct)
    if ce_strike is None or pe_strike is None:
        return None
    return [
        {"option_type": "CE", "strike": ce_strike, "action": "BUY", "qty": 1, "strategy_name": "Long Strangle"},
        {"option_type": "PE", "strike": pe_strike, "action": "BUY", "qty": 1, "strategy_name": "Long Strangle"},
    ]


# ══════════════════════════════════════════════════════════════════════════
#  VERTICAL SPREADS
# ══════════════════════════════════════════════════════════════════════════


def bull_call_spread(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    width_pct: float = 2.0,
):
    """Buy ATM Call, Sell OTM Call. Bullish, limited risk."""
    buy_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
    sell_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=width_pct)
    if buy_strike is None or sell_strike is None or buy_strike >= sell_strike:
        return None
    return [
        {"option_type": "CE", "strike": buy_strike, "action": "BUY", "qty": 1, "strategy_name": "Bull Call Spread"},
        {"option_type": "CE", "strike": sell_strike, "action": "SELL", "qty": 1, "strategy_name": "Bull Call Spread"},
    ]


def bear_put_spread(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    width_pct: float = 2.0,
):
    """Buy ATM Put, Sell OTM Put. Bearish, limited risk."""
    buy_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
    sell_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-width_pct)
    if buy_strike is None or sell_strike is None or buy_strike <= sell_strike:
        return None
    return [
        {"option_type": "PE", "strike": buy_strike, "action": "BUY", "qty": 1, "strategy_name": "Bear Put Spread"},
        {"option_type": "PE", "strike": sell_strike, "action": "SELL", "qty": 1, "strategy_name": "Bear Put Spread"},
    ]


def bull_put_spread(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    otm_pct: float = 2.0, width_pct: float = 2.0,
):
    """Sell OTM Put, Buy further OTM Put. Credit spread, bullish."""
    sell_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-otm_pct)
    buy_strike = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-(otm_pct + width_pct))
    if sell_strike is None or buy_strike is None or sell_strike <= buy_strike:
        return None
    return [
        {"option_type": "PE", "strike": sell_strike, "action": "SELL", "qty": 1, "strategy_name": "Bull Put Spread"},
        {"option_type": "PE", "strike": buy_strike, "action": "BUY", "qty": 1, "strategy_name": "Bull Put Spread"},
    ]


def bear_call_spread(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    otm_pct: float = 2.0, width_pct: float = 2.0,
):
    """Sell OTM Call, Buy further OTM Call. Credit spread, bearish."""
    sell_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct)
    buy_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct + width_pct)
    if sell_strike is None or buy_strike is None or sell_strike >= buy_strike:
        return None
    return [
        {"option_type": "CE", "strike": sell_strike, "action": "SELL", "qty": 1, "strategy_name": "Bear Call Spread"},
        {"option_type": "CE", "strike": buy_strike, "action": "BUY", "qty": 1, "strategy_name": "Bear Call Spread"},
    ]


# ══════════════════════════════════════════════════════════════════════════
#  MULTI-LEG STRATEGIES
# ══════════════════════════════════════════════════════════════════════════


def iron_condor(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    otm_pct: float = 3.0, wing_width_pct: float = 2.0,
):
    """
    Iron Condor: Sell OTM Call + OTM Put, Buy further OTM wings.
    Market-neutral, profit from time decay in a range.
    """
    sell_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct)
    buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct + wing_width_pct)
    sell_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-otm_pct)
    buy_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-(otm_pct + wing_width_pct))

    if any(s is None for s in [sell_ce, buy_ce, sell_pe, buy_pe]):
        return None
    if sell_ce >= buy_ce or sell_pe <= buy_pe:
        return None

    return [
        {"option_type": "CE", "strike": sell_ce, "action": "SELL", "qty": 1, "strategy_name": "Iron Condor"},
        {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "Iron Condor"},
        {"option_type": "PE", "strike": sell_pe, "action": "SELL", "qty": 1, "strategy_name": "Iron Condor"},
        {"option_type": "PE", "strike": buy_pe, "action": "BUY", "qty": 1, "strategy_name": "Iron Condor"},
    ]


def iron_butterfly(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    wing_width_pct: float = 3.0,
):
    """
    Iron Butterfly: Sell ATM Call + ATM Put, Buy OTM wings.
    Maximum profit at spot, higher premium than iron condor.
    """
    atm_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
    atm_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
    buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=wing_width_pct)
    buy_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-wing_width_pct)

    if any(s is None for s in [atm_ce, atm_pe, buy_ce, buy_pe]):
        return None

    return [
        {"option_type": "CE", "strike": atm_ce, "action": "SELL", "qty": 1, "strategy_name": "Iron Butterfly"},
        {"option_type": "PE", "strike": atm_pe, "action": "SELL", "qty": 1, "strategy_name": "Iron Butterfly"},
        {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "Iron Butterfly"},
        {"option_type": "PE", "strike": buy_pe, "action": "BUY", "qty": 1, "strategy_name": "Iron Butterfly"},
    ]


def jade_lizard(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    put_otm_pct: float = 2.0, call_otm_pct: float = 2.0, call_width_pct: float = 2.0,
):
    """
    Jade Lizard: Short Put + Short Call Spread (bear call).
    No upside risk if call credit covers the spread width.
    """
    sell_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-put_otm_pct)
    sell_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=call_otm_pct)
    buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=call_otm_pct + call_width_pct)

    if any(s is None for s in [sell_pe, sell_ce, buy_ce]):
        return None
    if sell_ce >= buy_ce:
        return None

    return [
        {"option_type": "PE", "strike": sell_pe, "action": "SELL", "qty": 1, "strategy_name": "Jade Lizard"},
        {"option_type": "CE", "strike": sell_ce, "action": "SELL", "qty": 1, "strategy_name": "Jade Lizard"},
        {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "Jade Lizard"},
    ]


def ratio_spread_call(
    engine, dt: date, chain: pd.DataFrame, spot: float, expiry: date,
    otm_pct: float = 3.0,
):
    """
    Call Ratio Spread: Buy 1 ATM Call, Sell 2 OTM Calls.
    Moderately bullish, profits from theta if market rises gently.
    """
    buy_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
    sell_strike = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=otm_pct)
    if buy_strike is None or sell_strike is None or buy_strike >= sell_strike:
        return None
    return [
        {"option_type": "CE", "strike": buy_strike, "action": "BUY", "qty": 1, "strategy_name": "Call Ratio Spread"},
        {"option_type": "CE", "strike": sell_strike, "action": "SELL", "qty": 2, "strategy_name": "Call Ratio Spread"},
    ]


# ══════════════════════════════════════════════════════════════════════════
#  STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════════════════

PREBUILT_STRATEGIES = {
    "Short Straddle": {
        "fn": short_straddle,
        "category": "Neutral",
        "description": "Sell ATM Call + Put. Max profit at spot. Unlimited risk.",
        "params": {},
    },
    "Long Straddle": {
        "fn": long_straddle,
        "category": "Volatility",
        "description": "Buy ATM Call + Put. Profit on large moves. Time decay hurts.",
        "params": {},
    },
    "Short Strangle": {
        "fn": lambda e, d, c, s, x: short_strangle(e, d, c, s, x, otm_pct=2.0),
        "category": "Neutral",
        "description": "Sell OTM Call + Put. Wider profit zone than straddle.",
        "params": {"otm_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "OTM %"}},
    },
    "Long Strangle": {
        "fn": lambda e, d, c, s, x: long_strangle(e, d, c, s, x, otm_pct=2.0),
        "category": "Volatility",
        "description": "Buy OTM Call + Put. Cheaper than straddle, needs bigger move.",
        "params": {"otm_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "OTM %"}},
    },
    "Bull Call Spread": {
        "fn": lambda e, d, c, s, x: bull_call_spread(e, d, c, s, x, width_pct=2.0),
        "category": "Bullish",
        "description": "Buy ATM Call, Sell OTM Call. Limited risk bullish bet.",
        "params": {"width_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Width %"}},
    },
    "Bear Put Spread": {
        "fn": lambda e, d, c, s, x: bear_put_spread(e, d, c, s, x, width_pct=2.0),
        "category": "Bearish",
        "description": "Buy ATM Put, Sell OTM Put. Limited risk bearish bet.",
        "params": {"width_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Width %"}},
    },
    "Bull Put Spread": {
        "fn": lambda e, d, c, s, x: bull_put_spread(e, d, c, s, x),
        "category": "Bullish",
        "description": "Sell OTM Put spread. Credit received, bullish bias.",
        "params": {
            "otm_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "OTM %"},
            "width_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Width %"},
        },
    },
    "Bear Call Spread": {
        "fn": lambda e, d, c, s, x: bear_call_spread(e, d, c, s, x),
        "category": "Bearish",
        "description": "Sell OTM Call spread. Credit received, bearish bias.",
        "params": {
            "otm_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "OTM %"},
            "width_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Width %"},
        },
    },
    "Iron Condor": {
        "fn": lambda e, d, c, s, x: iron_condor(e, d, c, s, x),
        "category": "Neutral",
        "description": "Sell OTM Call + Put spreads. Defined risk, range-bound.",
        "params": {
            "otm_pct": {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "OTM %"},
            "wing_width_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Wing Width %"},
        },
    },
    "Iron Butterfly": {
        "fn": lambda e, d, c, s, x: iron_butterfly(e, d, c, s, x),
        "category": "Neutral",
        "description": "Sell ATM straddle + buy wings. Higher credit, tighter range.",
        "params": {"wing_width_pct": {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "Wing Width %"}},
    },
    "Jade Lizard": {
        "fn": lambda e, d, c, s, x: jade_lizard(e, d, c, s, x),
        "category": "Bullish",
        "description": "Short Put + Short Call Spread. No upside risk if structured right.",
        "params": {
            "put_otm_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Put OTM %"},
            "call_otm_pct": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Call OTM %"},
        },
    },
    "Call Ratio Spread": {
        "fn": lambda e, d, c, s, x: ratio_spread_call(e, d, c, s, x),
        "category": "Bullish",
        "description": "Buy 1 ATM Call, Sell 2 OTM Calls. Moderately bullish.",
        "params": {"otm_pct": {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "OTM %"}},
    },
}


def get_strategy_fn(name: str, **params):
    """
    Get a strategy function by name, optionally with custom parameters.
    Returns a callable with signature (engine, dt, chain, spot, expiry).
    """
    entry = PREBUILT_STRATEGIES.get(name)
    if not entry:
        raise ValueError(f"Unknown strategy: {name}")
    
    base_fn = entry["fn"]
    if not params:
        return base_fn
    
    # Create a wrapper that passes custom params
    # Look up the original function (not the lambda) for param injection
    fn_map = {
        "Short Strangle": short_strangle,
        "Long Strangle": long_strangle,
        "Bull Call Spread": bull_call_spread,
        "Bear Put Spread": bear_put_spread,
        "Bull Put Spread": bull_put_spread,
        "Bear Call Spread": bear_call_spread,
        "Iron Condor": iron_condor,
        "Iron Butterfly": iron_butterfly,
        "Jade Lizard": jade_lizard,
        "Call Ratio Spread": ratio_spread_call,
    }
    
    original = fn_map.get(name)
    if original:
        return lambda e, d, c, s, x: original(e, d, c, s, x, **params)
    return base_fn
