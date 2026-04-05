"""
Price Action strategies for options trading.

Detects classic candlestick patterns and structural setups,
then generates options trade signals for the backtester.

Patterns detected:
  - Pin Bar (Hammer / Shooting Star)
  - Engulfing (Bullish / Bearish)
  - Inside Bar (compression → breakout)
  - Morning Star / Evening Star
  - Support/Resistance Breakout
  - Trend Pullback (EMA retest in trend)
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════
#  PATTERN DETECTION
# ══════════════════════════════════════════════════════════════════════════


def detect_pin_bars(df: pd.DataFrame, tail_ratio: float = 2.5) -> list[dict]:
    """
    Detect Pin Bars (hammers / shooting stars).
    Pin bar: long tail (wick) relative to body, small body.
    Bullish pin bar (hammer): long lower wick, price near high.
    Bearish pin bar (shooting star): long upper wick, price near low.
    """
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        full_range = h - l
        if full_range == 0 or body == 0:
            continue

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Bullish pin bar (hammer): lower wick > tail_ratio * body, upper wick small
        if lower_wick > tail_ratio * body and upper_wick < body:
            signals.append({
                "type": "PIN_BAR",
                "direction": "BULLISH",
                "idx": i,
                "date": row.get("date", row.name),
                "pattern": "Hammer",
                "close": c,
                "tail_size": round(lower_wick, 2),
            })

        # Bearish pin bar (shooting star): upper wick > tail_ratio * body, lower wick small
        if upper_wick > tail_ratio * body and lower_wick < body:
            signals.append({
                "type": "PIN_BAR",
                "direction": "BEARISH",
                "idx": i,
                "date": row.get("date", row.name),
                "pattern": "Shooting Star",
                "close": c,
                "tail_size": round(upper_wick, 2),
            })

    return signals


def detect_engulfing(df: pd.DataFrame) -> list[dict]:
    """
    Detect Bullish and Bearish Engulfing patterns.
    Bullish: bearish candle followed by larger bullish candle that engulfs it.
    Bearish: bullish candle followed by larger bearish candle that engulfs it.
    """
    signals = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_body = prev["close"] - prev["open"]
        curr_body = curr["close"] - curr["open"]

        # Bullish engulfing: prev bearish, curr bullish and engulfs
        if (prev_body < 0 and curr_body > 0
                and curr["open"] <= prev["close"]
                and curr["close"] >= prev["open"]):
            signals.append({
                "type": "ENGULFING",
                "direction": "BULLISH",
                "idx": i,
                "date": curr.get("date", curr.name),
                "close": curr["close"],
            })

        # Bearish engulfing: prev bullish, curr bearish and engulfs
        if (prev_body > 0 and curr_body < 0
                and curr["open"] >= prev["close"]
                and curr["close"] <= prev["open"]):
            signals.append({
                "type": "ENGULFING",
                "direction": "BEARISH",
                "idx": i,
                "date": curr.get("date", curr.name),
                "close": curr["close"],
            })

    return signals


def detect_inside_bars(df: pd.DataFrame) -> list[dict]:
    """
    Detect Inside Bar patterns (compression before breakout).
    Inside bar: current high < prev high AND current low > prev low.
    Signal comes on the bar AFTER the inside bar (breakout direction).
    """
    signals = []
    for i in range(2, len(df)):
        mother = df.iloc[i - 2]
        inside = df.iloc[i - 1]
        breakout = df.iloc[i]

        # Check if middle candle is an inside bar
        if inside["high"] < mother["high"] and inside["low"] > mother["low"]:
            # Breakout direction
            if breakout["close"] > mother["high"]:
                signals.append({
                    "type": "INSIDE_BAR",
                    "direction": "BULLISH",
                    "idx": i,
                    "date": breakout.get("date", breakout.name),
                    "mother_high": mother["high"],
                    "mother_low": mother["low"],
                    "close": breakout["close"],
                })
            elif breakout["close"] < mother["low"]:
                signals.append({
                    "type": "INSIDE_BAR",
                    "direction": "BEARISH",
                    "idx": i,
                    "date": breakout.get("date", breakout.name),
                    "mother_high": mother["high"],
                    "mother_low": mother["low"],
                    "close": breakout["close"],
                })

    return signals


def detect_morning_evening_star(df: pd.DataFrame, body_ratio: float = 0.3) -> list[dict]:
    """
    Detect Morning Star (bullish reversal) and Evening Star (bearish reversal).
    Morning Star: big bearish -> small body (doji) -> big bullish
    Evening Star: big bullish -> small body (doji) -> big bearish
    """
    signals = []
    for i in range(2, len(df)):
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]

        c1_body = abs(c1["close"] - c1["open"])
        c2_body = abs(c2["close"] - c2["open"])
        c3_body = abs(c3["close"] - c3["open"])
        c1_range = c1["high"] - c1["low"]
        c3_range = c3["high"] - c3["low"]

        if c1_range == 0 or c3_range == 0:
            continue

        is_small_middle = c2_body < c1_body * body_ratio

        # Morning Star
        if (c1["close"] < c1["open"]  # bearish
                and is_small_middle
                and c3["close"] > c3["open"]  # bullish
                and c3["close"] > (c1["open"] + c1["close"]) / 2):
            signals.append({
                "type": "STAR",
                "direction": "BULLISH",
                "idx": i,
                "date": c3.get("date", c3.name),
                "pattern": "Morning Star",
                "close": c3["close"],
            })

        # Evening Star
        if (c1["close"] > c1["open"]  # bullish
                and is_small_middle
                and c3["close"] < c3["open"]  # bearish
                and c3["close"] < (c1["open"] + c1["close"]) / 2):
            signals.append({
                "type": "STAR",
                "direction": "BEARISH",
                "idx": i,
                "date": c3.get("date", c3.name),
                "pattern": "Evening Star",
                "close": c3["close"],
            })

    return signals


def detect_support_resistance_breakout(df: pd.DataFrame, lookback: int = 20, threshold_pct: float = 0.3) -> list[dict]:
    """
    Detect breakouts above resistance or below support using recent swing points.
    """
    signals = []
    if len(df) < lookback + 1:
        return signals

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        curr = df.iloc[i]

        resistance = window["high"].max()
        support = window["low"].min()

        # Breakout above resistance
        if curr["close"] > resistance * (1 + threshold_pct / 100):
            signals.append({
                "type": "BREAKOUT",
                "direction": "BULLISH",
                "idx": i,
                "date": curr.get("date", curr.name),
                "level": round(resistance, 2),
                "close": curr["close"],
            })

        # Breakdown below support
        if curr["close"] < support * (1 - threshold_pct / 100):
            signals.append({
                "type": "BREAKOUT",
                "direction": "BEARISH",
                "idx": i,
                "date": curr.get("date", curr.name),
                "level": round(support, 2),
                "close": curr["close"],
            })

    return signals


def detect_ema_pullback(df: pd.DataFrame, ema_period: int = 20) -> list[dict]:
    """
    Detect trend pullbacks to EMA.
    In uptrend: price pulls back to EMA and bounces (bullish).
    In downtrend: price rallies to EMA and rejects (bearish).
    """
    if len(df) < ema_period + 5:
        return []

    df = df.copy()
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    signals = []

    for i in range(ema_period + 2, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        ema_val = ema.iloc[i]
        ema_prev = ema.iloc[i - 1]

        # Uptrend: price > EMA, and price recently touched/crossed EMA then bounced
        if curr["close"] > ema_val and ema_val > ema.iloc[i - 5]:  # EMA rising
            if prev["low"] <= ema_prev * 1.003:  # prev bar touched EMA
                if curr["close"] > prev["high"]:  # bounce confirmation
                    signals.append({
                        "type": "EMA_PULLBACK",
                        "direction": "BULLISH",
                        "idx": i,
                        "date": curr.get("date", curr.name),
                        "ema": round(ema_val, 2),
                        "close": curr["close"],
                    })

        # Downtrend: price < EMA, and price recently touched/crossed EMA then rejected
        if curr["close"] < ema_val and ema_val < ema.iloc[i - 5]:  # EMA falling
            if prev["high"] >= ema_prev * 0.997:  # prev bar touched EMA
                if curr["close"] < prev["low"]:  # rejection confirmation
                    signals.append({
                        "type": "EMA_PULLBACK",
                        "direction": "BEARISH",
                        "idx": i,
                        "date": curr.get("date", curr.name),
                        "ema": round(ema_val, 2),
                        "close": curr["close"],
                    })

    return signals


def run_price_action_analysis(df: pd.DataFrame) -> dict:
    """Run all price action pattern detections and return combined results."""
    pin_bars = detect_pin_bars(df)
    engulfing = detect_engulfing(df)
    inside_bars = detect_inside_bars(df)
    stars = detect_morning_evening_star(df)
    breakouts = detect_support_resistance_breakout(df)
    pullbacks = detect_ema_pullback(df)

    all_signals = pin_bars + engulfing + inside_bars + stars + breakouts + pullbacks
    all_signals.sort(key=lambda s: s["idx"])

    # Recent bias from last 5 signals
    recent = all_signals[-5:] if all_signals else []
    bull = sum(1 for s in recent if s["direction"] == "BULLISH")
    bear = sum(1 for s in recent if s["direction"] == "BEARISH")
    bias = "BULLISH" if bull > bear + 1 else "BEARISH" if bear > bull + 1 else "NEUTRAL"

    return {
        "pin_bars": pin_bars,
        "engulfing": engulfing,
        "inside_bars": inside_bars,
        "stars": stars,
        "breakouts": breakouts,
        "ema_pullbacks": pullbacks,
        "all_signals": all_signals,
        "bias": bias,
    }


# ══════════════════════════════════════════════════════════════════════════
#  PRICE ACTION OPTIONS STRATEGIES FOR BACKTESTER
# ══════════════════════════════════════════════════════════════════════════


def _get_recent_ohlc(engine, dt: date, lookback: int = 50) -> Optional[pd.DataFrame]:
    """Extract recent underlying OHLC from the options data."""
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
    if len(rows) < 10:
        return None
    return pd.DataFrame(rows)


def pa_pin_bar_reversal(engine, dt, chain, spot, expiry):
    """
    Price Action: Trade reversals on pin bar (hammer/shooting star) patterns.
    Bullish hammer -> Bull Call Spread.
    Bearish shooting star -> Bear Put Spread.
    """
    ohlc = _get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    pin_bars = detect_pin_bars(ohlc)
    if not pin_bars:
        return None

    latest = pin_bars[-1]
    if latest["idx"] < len(ohlc) - 2:
        return None

    if latest["direction"] == "BULLISH":
        buy = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.0)
        if buy and sell and buy < sell:
            return [
                {"option_type": "CE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA Pin Bar Reversal"},
                {"option_type": "CE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA Pin Bar Reversal"},
            ]
    elif latest["direction"] == "BEARISH":
        buy = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.0)
        if buy and sell and buy > sell:
            return [
                {"option_type": "PE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA Pin Bar Reversal"},
                {"option_type": "PE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA Pin Bar Reversal"},
            ]
    return None


def pa_engulfing_momentum(engine, dt, chain, spot, expiry):
    """
    Price Action: Trade engulfing candle momentum.
    Bullish engulfing -> Bull Call Spread.
    Bearish engulfing -> Bear Put Spread.
    """
    ohlc = _get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    engulfing = detect_engulfing(ohlc)
    if not engulfing:
        return None

    latest = engulfing[-1]
    if latest["idx"] < len(ohlc) - 2:
        return None

    if latest["direction"] == "BULLISH":
        buy = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=2.5)
        if buy and sell and buy < sell:
            return [
                {"option_type": "CE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA Engulfing"},
                {"option_type": "CE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA Engulfing"},
            ]
    elif latest["direction"] == "BEARISH":
        buy = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-2.5)
        if buy and sell and buy > sell:
            return [
                {"option_type": "PE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA Engulfing"},
                {"option_type": "PE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA Engulfing"},
            ]
    return None


def pa_inside_bar_breakout(engine, dt, chain, spot, expiry):
    """
    Price Action: Trade inside bar breakouts (compression → expansion).
    Bullish breakout -> Bull Call Spread.
    Bearish breakdown -> Bear Put Spread.
    """
    ohlc = _get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    inside_bars = detect_inside_bars(ohlc)
    if not inside_bars:
        return None

    latest = inside_bars[-1]
    if latest["idx"] < len(ohlc) - 2:
        return None

    if latest["direction"] == "BULLISH":
        buy = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.0)
        if buy and sell and buy < sell:
            return [
                {"option_type": "CE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA Inside Bar Breakout"},
                {"option_type": "CE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA Inside Bar Breakout"},
            ]
    elif latest["direction"] == "BEARISH":
        buy = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.0)
        if buy and sell and buy > sell:
            return [
                {"option_type": "PE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA Inside Bar Breakout"},
                {"option_type": "PE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA Inside Bar Breakout"},
            ]
    return None


def pa_sr_breakout(engine, dt, chain, spot, expiry):
    """
    Price Action: Trade support/resistance breakouts with momentum.
    Break above resistance -> Bull Call Spread.
    Break below support -> Bear Put Spread.
    """
    ohlc = _get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    breakouts = detect_support_resistance_breakout(ohlc)
    if not breakouts:
        return None

    latest = breakouts[-1]
    if latest["idx"] < len(ohlc) - 2:
        return None

    if latest["direction"] == "BULLISH":
        buy = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=3.0)
        if buy and sell and buy < sell:
            return [
                {"option_type": "CE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA S/R Breakout"},
                {"option_type": "CE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA S/R Breakout"},
            ]
    elif latest["direction"] == "BEARISH":
        buy = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-3.0)
        if buy and sell and buy > sell:
            return [
                {"option_type": "PE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA S/R Breakout"},
                {"option_type": "PE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA S/R Breakout"},
            ]
    return None


def pa_ema_pullback_trend(engine, dt, chain, spot, expiry):
    """
    Price Action: Trade EMA pullbacks in trending markets.
    Bullish pullback bounce -> Bull Call Spread.
    Bearish pullback rejection -> Bear Put Spread.
    """
    ohlc = _get_recent_ohlc(engine, dt)
    if ohlc is None:
        return None

    pullbacks = detect_ema_pullback(ohlc, ema_period=20)
    if not pullbacks:
        return None

    latest = pullbacks[-1]
    if latest["idx"] < len(ohlc) - 2:
        return None

    if latest["direction"] == "BULLISH":
        buy = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "CE", offset_pct=2.5)
        if buy and sell and buy < sell:
            return [
                {"option_type": "CE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA EMA Pullback"},
                {"option_type": "CE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA EMA Pullback"},
            ]
    elif latest["direction"] == "BEARISH":
        buy = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=0)
        sell = engine.find_strike_near_spot(chain, spot, "PE", offset_pct=-2.5)
        if buy and sell and buy > sell:
            return [
                {"option_type": "PE", "strike": buy, "action": "BUY", "qty": 1, "strategy_name": "PA EMA Pullback"},
                {"option_type": "PE", "strike": sell, "action": "SELL", "qty": 1, "strategy_name": "PA EMA Pullback"},
            ]
    return None


# ══════════════════════════════════════════════════════════════════════════
#  STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════════════════

PA_STRATEGIES = {
    "PA Pin Bar Reversal": {
        "fn": pa_pin_bar_reversal,
        "category": "Price Action",
        "description": "Trade reversals on hammer/shooting star pin bar patterns.",
    },
    "PA Engulfing Momentum": {
        "fn": pa_engulfing_momentum,
        "category": "Price Action",
        "description": "Trade momentum after bullish/bearish engulfing candles.",
    },
    "PA Inside Bar Breakout": {
        "fn": pa_inside_bar_breakout,
        "category": "Price Action",
        "description": "Trade breakouts from inside bar compression patterns.",
    },
    "PA S/R Breakout": {
        "fn": pa_sr_breakout,
        "category": "Price Action",
        "description": "Trade breakouts above resistance or below support.",
    },
    "PA EMA Pullback": {
        "fn": pa_ema_pullback_trend,
        "category": "Price Action",
        "description": "Trade pullbacks to 20-EMA in trending markets.",
    },
}
