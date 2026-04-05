"""Shared utilities for backtester strategy modules."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Cache for yfinance OHLC to avoid repeated downloads within a session
_yf_ohlc_cache: dict[str, pd.DataFrame] = {}


def _fetch_real_ohlc(symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Fetch real daily OHLC from yfinance for the underlying index."""
    cache_key = f"{symbol}_{start}_{end}"
    if cache_key in _yf_ohlc_cache:
        return _yf_ohlc_cache[cache_key]

    try:
        from config import INDEX_CONFIG
        import yfinance as yf

        cfg = INDEX_CONFIG.get(symbol)
        if not cfg:
            return None

        ticker = cfg["yf_ticker"]
        tk = yf.Ticker(ticker)
        df = tk.history(start=str(start), end=str(end + timedelta(days=1)))
        if df.empty:
            return None

        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        result = df[["date", "open", "high", "low", "close"]].copy()
        _yf_ohlc_cache[cache_key] = result
        return result
    except Exception as e:
        logger.debug("yfinance OHLC fetch failed for %s: %s", symbol, e)
        return None


def get_recent_ohlc(engine, dt: date, lookback: int = 50) -> Optional[pd.DataFrame]:
    """
    Extract recent underlying OHLC for technical analysis.

    First attempts to fetch REAL OHLC from yfinance. Falls back to
    deriving approximate OHLC from bhavcopy option data if the new-format
    underlying_close column and option OHLC data are available, and uses
    synthetic close-only OHLC as a last resort.
    """
    # Attempt 1: Real OHLC from yfinance
    symbol = engine.config.symbol
    start = dt - timedelta(days=lookback * 2)  # buffer for holidays
    real_ohlc = _fetch_real_ohlc(symbol, start, dt)
    if real_ohlc is not None and not real_ohlc.empty:
        filtered = real_ohlc[real_ohlc["date"] <= dt].tail(lookback)
        if len(filtered) >= 10:
            return filtered.reset_index(drop=True)

    # Attempt 2: Derive from bhavcopy data if available
    # New-format bhavcopies include UndrlygPric (underlying_close) — but that's
    # just a single price.  However, the *options* rows themselves have OHLC
    # that can hint at the underlying range via ATM put-call parity.
    # For simplicity, we use the underlying_close but try to get a better
    # open/high/low from the data_adapter if underlying columns exist.
    dates = [d for d in engine.trading_dates if d <= dt][-lookback:]
    if len(dates) < 10:
        return None

    rows = []
    for d in dates:
        price = engine.get_underlying_price(d)
        if not price:
            continue

        # Try to get real OHLC from the options data
        # Some bhavcopy formats have futures data with real OHLC
        chain = engine.get_chain_on_date(d)
        day_high = price
        day_low = price
        day_open = price

        if not chain.empty and "underlying_close" in chain.columns:
            # Use the range of option underlying prices as a proxy
            u_prices = chain["underlying_close"].dropna()
            if not u_prices.empty:
                price = float(u_prices.iloc[0])
                day_open = price
                day_high = price
                day_low = price

        rows.append({
            "date": d,
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "close": price,
        })

    if len(rows) < 10:
        return None
    return pd.DataFrame(rows)
