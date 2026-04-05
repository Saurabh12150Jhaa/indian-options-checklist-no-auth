"""
Smart Money Concepts (SMC) strategies for options trading.

Implements ICT / institutional-style analysis:
  - Order Blocks (OB): Last bearish candle before a bullish impulse (bullish OB) and vice versa
  - Fair Value Gaps (FVG): Price imbalances (gaps between candle 1 high and candle 3 low)
  - Break of Structure (BOS): Higher high (bullish) or lower low (bearish) confirming trend
  - Change of Character (CHoCH): First break against the trend (potential reversal)
  - Liquidity Sweeps: Price takes out a swing high/low then reverses (stop hunts)
  - Premium/Discount Zones: Based on equilibrium of recent range

Each detector returns a signal dict; strategy functions combine them
into options trade signals for the backtester.
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from backtester.utils import get_recent_ohlc


# ══════════════════════════════════════════════════════════════════════════
#  STRUCTURE DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Identify swing highs and swing lows in OHLC data.
    A swing high: high[i] > all highs in [i-lookback, i+lookback]
    A swing low: low[i] < all lows in [i-lookback, i+lookback]
    """
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False

    highs = df["high"].values
    lows = df["low"].values

    for i in range(lookback, len(df) - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        window_low = lows[i - lookback : i + lookback + 1]
        if highs[i] == max(window_high):
            df.iloc[i, df.columns.get_loc("swing_high")] = True
        if lows[i] == min(window_low):
            df.iloc[i, df.columns.get_loc("swing_low")] = True

    return df


def detect_bos(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """
    Detect Break of Structure (BOS).
    Bullish BOS: price breaks above a previous swing high
    Bearish BOS: price breaks below a previous swing low
    """
    df = find_swing_points(df, lookback)
    signals = []
    last_swing_high = None
    last_swing_low = None

    for i in range(len(df)):
        row = df.iloc[i]
        if row["swing_high"]:
            last_swing_high = {"price": row["high"], "idx": i}
        if row["swing_low"]:
            last_swing_low = {"price": row["low"], "idx": i}

        # Bullish BOS: close above last swing high
        if last_swing_high and row["close"] > last_swing_high["price"] and i > last_swing_high["idx"]:
            signals.append({
                "type": "BOS",
                "direction": "BULLISH",
                "idx": i,
                "date": row.get("date", row.name),
                "price": last_swing_high["price"],
                "close": row["close"],
            })
            last_swing_high = None  # consumed

        # Bearish BOS: close below last swing low
        if last_swing_low and row["close"] < last_swing_low["price"] and i > last_swing_low["idx"]:
            signals.append({
                "type": "BOS",
                "direction": "BEARISH",
                "idx": i,
                "date": row.get("date", row.name),
                "price": last_swing_low["price"],
                "close": row["close"],
            })
            last_swing_low = None

    return signals


def detect_choch(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """
    Detect Change of Character (CHoCH) — the first break against the prevailing trend.
    In an uptrend (series of higher highs), CHoCH = first lower low.
    In a downtrend (series of lower lows), CHoCH = first higher high.
    """
    df = find_swing_points(df, lookback)
    signals = []
    swing_highs = []
    swing_lows = []
    trend = None  # 'up' or 'down'

    for i in range(len(df)):
        row = df.iloc[i]

        if row["swing_high"]:
            if swing_highs and row["high"] > swing_highs[-1]["price"]:
                if trend != "up":
                    trend = "up"
            elif swing_highs and row["high"] < swing_highs[-1]["price"] and trend == "up":
                signals.append({
                    "type": "CHoCH",
                    "direction": "BEARISH",
                    "idx": i,
                    "date": row.get("date", row.name),
                    "price": row["high"],
                    "prev_high": swing_highs[-1]["price"],
                })
                trend = "down"
            swing_highs.append({"price": row["high"], "idx": i})

        if row["swing_low"]:
            if swing_lows and row["low"] < swing_lows[-1]["price"]:
                if trend != "down":
                    trend = "down"
            elif swing_lows and row["low"] > swing_lows[-1]["price"] and trend == "down":
                signals.append({
                    "type": "CHoCH",
                    "direction": "BULLISH",
                    "idx": i,
                    "date": row.get("date", row.name),
                    "price": row["low"],
                    "prev_low": swing_lows[-1]["price"],
                })
                trend = "up"
            swing_lows.append({"price": row["low"], "idx": i})

    return signals


def detect_order_blocks(df: pd.DataFrame, min_impulse_pct: float = 0.5) -> list[dict]:
    """
    Detect Order Blocks (OB).
    Bullish OB: Last bearish candle before a strong bullish impulse move.
    Bearish OB: Last bullish candle before a strong bearish impulse move.
    """
    signals = []
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]

        body_pct = abs(curr["close"] - curr["open"]) / curr["open"] * 100 if curr["open"] > 0 else 0

        # Bullish OB: prev is bearish, curr+next form strong bullish impulse
        if prev["close"] < prev["open"]:  # bearish candle
            impulse = (nxt["close"] - prev["low"]) / prev["low"] * 100 if prev["low"] > 0 else 0
            if impulse >= min_impulse_pct:
                signals.append({
                    "type": "ORDER_BLOCK",
                    "direction": "BULLISH",
                    "idx": i - 1,
                    "date": prev.get("date", prev.name),
                    "ob_high": prev["open"],
                    "ob_low": prev["close"],
                    "impulse_pct": round(impulse, 2),
                })

        # Bearish OB: prev is bullish, curr+next form strong bearish impulse
        if prev["close"] > prev["open"]:  # bullish candle
            impulse = (prev["high"] - nxt["close"]) / prev["high"] * 100 if prev["high"] > 0 else 0
            if impulse >= min_impulse_pct:
                signals.append({
                    "type": "ORDER_BLOCK",
                    "direction": "BEARISH",
                    "idx": i - 1,
                    "date": prev.get("date", prev.name),
                    "ob_high": prev["close"],
                    "ob_low": prev["open"],
                    "impulse_pct": round(impulse, 2),
                })

    return signals


def detect_fvg(df: pd.DataFrame) -> list[dict]:
    """
    Detect Fair Value Gaps (FVG) — imbalances in price delivery.
    Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
    """
    signals = []
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        nxt = df.iloc[i + 1]

        # Bullish FVG
        if prev["high"] < nxt["low"]:
            signals.append({
                "type": "FVG",
                "direction": "BULLISH",
                "idx": i,
                "date": df.iloc[i].get("date", df.iloc[i].name),
                "gap_top": nxt["low"],
                "gap_bottom": prev["high"],
                "gap_size": round(nxt["low"] - prev["high"], 2),
            })

        # Bearish FVG
        if prev["low"] > nxt["high"]:
            signals.append({
                "type": "FVG",
                "direction": "BEARISH",
                "idx": i,
                "date": df.iloc[i].get("date", df.iloc[i].name),
                "gap_top": prev["low"],
                "gap_bottom": nxt["high"],
                "gap_size": round(prev["low"] - nxt["high"], 2),
            })

    return signals


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10, reversal_pct: float = 0.3) -> list[dict]:
    """
    Detect Liquidity Sweeps (stop hunts).
    Price briefly takes out a swing high/low (where stops cluster),
    then reverses sharply.
    """
    df_with_swings = find_swing_points(df, lookback=lookback)
    signals = []

    swing_highs = []
    swing_lows = []

    for i in range(len(df_with_swings)):
        row = df_with_swings.iloc[i]

        if row["swing_high"]:
            swing_highs.append({"price": row["high"], "idx": i})
        if row["swing_low"]:
            swing_lows.append({"price": row["low"], "idx": i})

        # Check for sweep of recent swing highs (bearish sweep)
        for sh in swing_highs[-5:]:
            if row["high"] > sh["price"] and row["close"] < sh["price"]:
                reversal = (row["high"] - row["close"]) / row["high"] * 100 if row["high"] > 0 else 0
                if reversal >= reversal_pct:
                    signals.append({
                        "type": "LIQUIDITY_SWEEP",
                        "direction": "BEARISH",
                        "idx": i,
                        "date": row.get("date", row.name),
                        "swept_level": sh["price"],
                        "high": row["high"],
                        "close": row["close"],
                        "reversal_pct": round(reversal, 2),
                    })

        # Check for sweep of recent swing lows (bullish sweep)
        for sl in swing_lows[-5:]:
            if row["low"] < sl["price"] and row["close"] > sl["price"]:
                reversal = (row["close"] - row["low"]) / row["close"] * 100 if row["close"] > 0 else 0
                if reversal >= reversal_pct:
                    signals.append({
                        "type": "LIQUIDITY_SWEEP",
                        "direction": "BULLISH",
                        "idx": i,
                        "date": row.get("date", row.name),
                        "swept_level": sl["price"],
                        "low": row["low"],
                        "close": row["close"],
                        "reversal_pct": round(reversal, 2),
                    })

    return signals


def premium_discount_zone(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Calculate premium/discount zones based on recent range equilibrium.
    Premium zone: above 50% of range (overvalued, look for sells)
    Discount zone: below 50% of range (undervalued, look for buys)
    """
    recent = df.tail(lookback)
    if recent.empty:
        return {"equilibrium": 0, "premium_zone": 0, "discount_zone": 0, "zone": "NEUTRAL"}

    high = recent["high"].max()
    low = recent["low"].min()
    eq = (high + low) / 2
    current = df.iloc[-1]["close"]

    zone = "PREMIUM" if current > eq else "DISCOUNT" if current < eq else "EQUILIBRIUM"

    return {
        "high": round(high, 2),
        "low": round(low, 2),
        "equilibrium": round(eq, 2),
        "premium_zone": round(eq + (high - eq) * 0.5, 2),
        "discount_zone": round(low + (eq - low) * 0.5, 2),
        "current": round(current, 2),
        "zone": zone,
    }


def run_smc_analysis(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Run full SMC analysis on OHLC data.
    Returns all detected signals and the current market context.
    """
    bos_signals = detect_bos(df, lookback)
    choch_signals = detect_choch(df, lookback)
    ob_signals = detect_order_blocks(df)
    fvg_signals = detect_fvg(df)
    sweep_signals = detect_liquidity_sweep(df, lookback)
    pd_zone = premium_discount_zone(df, lookback=20)

    # Determine overall SMC bias from recent signals
    recent_bos = [s for s in bos_signals[-3:]] if bos_signals else []
    recent_choch = [s for s in choch_signals[-2:]] if choch_signals else []
    recent_sweeps = [s for s in sweep_signals[-2:]] if sweep_signals else []

    bullish_score = sum(1 for s in recent_bos if s["direction"] == "BULLISH")
    bearish_score = sum(1 for s in recent_bos if s["direction"] == "BEARISH")
    bullish_score += sum(1 for s in recent_choch if s["direction"] == "BULLISH") * 2
    bearish_score += sum(1 for s in recent_choch if s["direction"] == "BEARISH") * 2
    bullish_score += sum(1 for s in recent_sweeps if s["direction"] == "BULLISH")
    bearish_score += sum(1 for s in recent_sweeps if s["direction"] == "BEARISH")

    if pd_zone["zone"] == "DISCOUNT":
        bullish_score += 1
    elif pd_zone["zone"] == "PREMIUM":
        bearish_score += 1

    if bullish_score > bearish_score + 1:
        bias = "BULLISH"
    elif bearish_score > bullish_score + 1:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bos": bos_signals,
        "choch": choch_signals,
        "order_blocks": ob_signals,
        "fvg": fvg_signals,
        "liquidity_sweeps": sweep_signals,
        "premium_discount": pd_zone,
        "bias": bias,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
    }


# ══════════════════════════════════════════════════════════════════════════
#  SMC-BASED OPTIONS STRATEGIES FOR BACKTESTER
# ══════════════════════════════════════════════════════════════════════════


def smc_bullish_ob_entry(engine, dt, chain, spot, expiry):
    """
    SMC Strategy: Enter bullish trades when price retests a bullish Order Block
    in the discount zone, confirmed by bullish BOS.
    Trade: Bull Put Spread (credit, bullish).
    """
    ohlc = get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    smc = run_smc_analysis(ohlc)

    if smc["bias"] != "BULLISH":
        return None
    if smc["premium_discount"]["zone"] != "DISCOUNT":
        return None
    if not smc["order_blocks"]:
        return None

    # Look for a recent bullish OB that price is near
    bullish_obs = [ob for ob in smc["order_blocks"] if ob["direction"] == "BULLISH"]
    if not bullish_obs:
        return None

    latest_ob = bullish_obs[-1]
    # Price should be near the OB zone
    if not (latest_ob["ob_low"] * 0.99 <= spot <= latest_ob["ob_high"] * 1.02):
        return None

    # Enter Bull Put Spread
    sell_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-1.5)
    buy_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.5)
    if sell_pe is None or buy_pe is None or sell_pe <= buy_pe:
        return None
    return [
        {"option_type": "PE", "strike": sell_pe, "action": "SELL", "qty": 1, "strategy_name": "SMC Bullish OB"},
        {"option_type": "PE", "strike": buy_pe, "action": "BUY", "qty": 1, "strategy_name": "SMC Bullish OB"},
    ]


def smc_bearish_ob_entry(engine, dt, chain, spot, expiry):
    """
    SMC Strategy: Enter bearish trades when price retests a bearish Order Block
    in the premium zone, confirmed by bearish BOS.
    Trade: Bear Call Spread (credit, bearish).
    """
    ohlc = get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    smc = run_smc_analysis(ohlc)

    if smc["bias"] != "BEARISH":
        return None
    if smc["premium_discount"]["zone"] != "PREMIUM":
        return None
    if not smc["order_blocks"]:
        return None

    bearish_obs = [ob for ob in smc["order_blocks"] if ob["direction"] == "BEARISH"]
    if not bearish_obs:
        return None

    latest_ob = bearish_obs[-1]
    if not (latest_ob["ob_low"] * 0.98 <= spot <= latest_ob["ob_high"] * 1.01):
        return None

    sell_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=1.5)
    buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.5)
    if sell_ce is None or buy_ce is None or sell_ce >= buy_ce:
        return None
    return [
        {"option_type": "CE", "strike": sell_ce, "action": "SELL", "qty": 1, "strategy_name": "SMC Bearish OB"},
        {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "SMC Bearish OB"},
    ]


def smc_liquidity_sweep_reversal(engine, dt, chain, spot, expiry):
    """
    SMC Strategy: Trade reversals after liquidity sweeps.
    After a bearish sweep (stops taken below swing low, then reversal up) -> Buy Call.
    After a bullish sweep (stops taken above swing high, then reversal down) -> Buy Put.
    """
    ohlc = get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    smc = run_smc_analysis(ohlc)
    sweeps = smc["liquidity_sweeps"]
    if not sweeps:
        return None

    latest = sweeps[-1]
    # Only act on very recent sweeps (within last 2 bars)
    if latest["idx"] < len(ohlc) - 3:
        return None

    if latest["direction"] == "BULLISH":
        # Bullish sweep reversal -> buy call spread
        buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.0)
        if buy_ce is None or sell_ce is None or buy_ce >= sell_ce:
            return None
        return [
            {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "SMC Sweep Reversal"},
            {"option_type": "CE", "strike": sell_ce, "action": "SELL", "qty": 1, "strategy_name": "SMC Sweep Reversal"},
        ]

    elif latest["direction"] == "BEARISH":
        buy_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.0)
        if buy_pe is None or sell_pe is None or buy_pe <= sell_pe:
            return None
        return [
            {"option_type": "PE", "strike": buy_pe, "action": "BUY", "qty": 1, "strategy_name": "SMC Sweep Reversal"},
            {"option_type": "PE", "strike": sell_pe, "action": "SELL", "qty": 1, "strategy_name": "SMC Sweep Reversal"},
        ]

    return None


def smc_fvg_fill(engine, dt, chain, spot, expiry):
    """
    SMC Strategy: Trade FVG fills — price tends to return to fill gaps.
    Bullish FVG below price -> wait for pullback into gap, then buy calls.
    Bearish FVG above price -> wait for rally into gap, then buy puts.
    """
    ohlc = get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    smc = run_smc_analysis(ohlc)
    fvgs = smc["fvg"]
    if not fvgs:
        return None

    # Check for unfilled FVGs near current price
    for fvg in reversed(fvgs[-10:]):
        if fvg["direction"] == "BULLISH":
            # Bullish FVG: gap is below. If price has pulled back into the gap -> buy
            if fvg["gap_bottom"] <= spot <= fvg["gap_top"]:
                buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
                sell_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.0)
                if buy_ce and sell_ce and buy_ce < sell_ce:
                    return [
                        {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "SMC FVG Fill"},
                        {"option_type": "CE", "strike": sell_ce, "action": "SELL", "qty": 1, "strategy_name": "SMC FVG Fill"},
                    ]

        elif fvg["direction"] == "BEARISH":
            if fvg["gap_bottom"] <= spot <= fvg["gap_top"]:
                buy_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
                sell_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.0)
                if buy_pe and sell_pe and buy_pe > sell_pe:
                    return [
                        {"option_type": "PE", "strike": buy_pe, "action": "BUY", "qty": 1, "strategy_name": "SMC FVG Fill"},
                        {"option_type": "PE", "strike": sell_pe, "action": "SELL", "qty": 1, "strategy_name": "SMC FVG Fill"},
                    ]

    return None


def smc_choch_trend_reversal(engine, dt, chain, spot, expiry):
    """
    SMC Strategy: Trade Change of Character (CHoCH) reversals.
    When market structure shifts (CHoCH), enter in the new direction using spreads.
    """
    ohlc = get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    smc = run_smc_analysis(ohlc)
    choch = smc["choch"]
    if not choch:
        return None

    latest = choch[-1]
    # Only act on recent CHoCH (within last 3 bars)
    if latest["idx"] < len(ohlc) - 4:
        return None

    if latest["direction"] == "BULLISH":
        # Structure shifted bullish -> Bull Call Spread
        buy_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell_ce = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.0)
        if buy_ce and sell_ce and buy_ce < sell_ce:
            return [
                {"option_type": "CE", "strike": buy_ce, "action": "BUY", "qty": 1, "strategy_name": "SMC CHoCH Reversal"},
                {"option_type": "CE", "strike": sell_ce, "action": "SELL", "qty": 1, "strategy_name": "SMC CHoCH Reversal"},
            ]
    elif latest["direction"] == "BEARISH":
        buy_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell_pe = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.0)
        if buy_pe and sell_pe and buy_pe > sell_pe:
            return [
                {"option_type": "PE", "strike": buy_pe, "action": "BUY", "qty": 1, "strategy_name": "SMC CHoCH Reversal"},
                {"option_type": "PE", "strike": sell_pe, "action": "SELL", "qty": 1, "strategy_name": "SMC CHoCH Reversal"},
            ]

    return None


# ══════════════════════════════════════════════════════════════════════════
#  STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════════════════

SMC_STRATEGIES = {
    "SMC Bullish Order Block": {
        "fn": smc_bullish_ob_entry,
        "category": "Smart Money",
        "description": "Enter bullish spread when price retests a bullish Order Block in discount zone with BOS confirmation.",
    },
    "SMC Bearish Order Block": {
        "fn": smc_bearish_ob_entry,
        "category": "Smart Money",
        "description": "Enter bearish spread when price retests a bearish Order Block in premium zone with BOS confirmation.",
    },
    "SMC Liquidity Sweep": {
        "fn": smc_liquidity_sweep_reversal,
        "category": "Smart Money",
        "description": "Trade reversals after stop hunts / liquidity sweeps beyond swing points.",
    },
    "SMC FVG Fill": {
        "fn": smc_fvg_fill,
        "category": "Smart Money",
        "description": "Enter when price pulls back to fill a Fair Value Gap (imbalance).",
    },
    "SMC CHoCH Reversal": {
        "fn": smc_choch_trend_reversal,
        "category": "Smart Money",
        "description": "Trade the first structural shift (Change of Character) indicating trend reversal.",
    },
}
