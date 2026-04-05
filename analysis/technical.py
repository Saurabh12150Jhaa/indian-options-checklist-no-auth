"""Technical analysis indicators: EMA, RSI, MACD, pivot points, Fibonacci retracement."""

from typing import Optional

import pandas as pd

from config import EMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL


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
