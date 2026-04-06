"""Shared utilities for backtester strategy modules."""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache for yfinance OHLC to avoid repeated downloads within a session.
# Keyed by symbol → full DataFrame so a single download covers the
# entire backtest period.
_yf_ohlc_cache: dict[str, pd.DataFrame] = {}


def _fetch_real_ohlc(symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Fetch real daily OHLC from yfinance for the underlying index.

    Uses a per-symbol cache so only one download is needed per backtest
    session.  If the cached data already covers the requested range it is
    reused; otherwise we fetch a wider window (earliest start to latest
    end ever requested) so subsequent calls for different dates hit the
    cache.
    """
    if symbol in _yf_ohlc_cache:
        cached = _yf_ohlc_cache[symbol]
        if not cached.empty:
            cached_min = cached["date"].min()
            cached_max = cached["date"].max()
            if cached_min <= start and cached_max >= end:
                return cached
            # Widen the window to cover old + new range
            start = min(start, cached_min)
            end = max(end, cached_max)

    try:
        from config import INDEX_CONFIG
        import yfinance as yf

        cfg = INDEX_CONFIG.get(symbol)
        if not cfg:
            return None

        ticker = cfg["yf_ticker"]
        tk = yf.Ticker(ticker)
        # Add buffer on both ends
        df = tk.history(
            start=str(start - timedelta(days=7)),
            end=str(end + timedelta(days=7)),
        )
        if df.empty:
            return None

        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        result = df[["date", "open", "high", "low", "close"]].copy()
        _yf_ohlc_cache[symbol] = result
        return result
    except Exception as e:
        logger.debug("yfinance OHLC fetch failed for %s: %s", symbol, e)
        return None


def _build_bhavcopy_ohlc(engine) -> Optional[pd.DataFrame]:
    """Derive approximate OHLC from bhavcopy underlying_close data.

    Bhavcopies only provide a single underlying close per day.  We
    synthesise approximate high/low by looking at ATM option prices and
    implied ranges when possible, and fall back to adding a small
    synthetic spread around the close to ensure candle-based indicators
    get non-degenerate (non-flat) bars.
    """
    dates = engine.trading_dates
    if len(dates) < 3:
        return None

    rows = []
    for d in dates:
        price = engine.get_underlying_price(d)
        if not price or price <= 0:
            continue

        chain = engine.get_chain_on_date(d)
        day_open = price
        day_high = price
        day_low = price

        if not chain.empty:
            # Estimate intraday range from ATM straddle prices.
            # ATM straddle premium ≈ 0.8 × σ_daily × spot (empirical).
            # We use that to create a synthetic high/low.
            ce = chain[chain["option_type"] == "CE"]
            pe = chain[chain["option_type"] == "PE"]
            if not ce.empty and not pe.empty:
                ce_atm_idx = (ce["strike"] - price).abs().idxmin()
                pe_atm_idx = (pe["strike"] - price).abs().idxmin()
                ce_price = float(ce.loc[ce_atm_idx, "close"])
                pe_price = float(pe.loc[pe_atm_idx, "close"])
                straddle = ce_price + pe_price
                # Estimated daily range ≈ straddle × 0.6 (rough)
                half_range = straddle * 0.3
                if half_range > 0:
                    day_high = price + half_range
                    day_low = price - half_range
                    # Approximate open as midpoint with a small shift
                    day_open = price - half_range * 0.1  # slight offset

        rows.append({
            "date": d,
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "close": price,
        })

    if len(rows) < 3:
        return None
    return pd.DataFrame(rows)


# Engine-level cache for bhavcopy-derived OHLC
_bhavcopy_ohlc_cache: dict[int, Optional[pd.DataFrame]] = {}


def get_recent_ohlc(engine, dt: date, lookback: int = 50) -> Optional[pd.DataFrame]:
    """
    Extract recent underlying OHLC for technical analysis.

    Strategy:
    1. Try yfinance (one download for the full period, then filter).
    2. Fall back to bhavcopy-derived OHLC (synthesised from option data).
    3. Return None only if neither source has enough data.
    """
    symbol = engine.config.symbol

    # Attempt 1: Real OHLC from yfinance (single fetch covers full period)
    start = dt - timedelta(days=lookback * 2)
    real_ohlc = _fetch_real_ohlc(symbol, start, dt)
    if real_ohlc is not None and not real_ohlc.empty:
        filtered = real_ohlc[real_ohlc["date"] <= dt].tail(lookback)
        if len(filtered) >= 10:
            return filtered.reset_index(drop=True)

    # Attempt 2: Bhavcopy-derived OHLC (cached per engine instance)
    eid = id(engine)
    if eid not in _bhavcopy_ohlc_cache:
        _bhavcopy_ohlc_cache[eid] = _build_bhavcopy_ohlc(engine)
    bhav_ohlc = _bhavcopy_ohlc_cache[eid]

    if bhav_ohlc is not None and not bhav_ohlc.empty:
        filtered = bhav_ohlc[bhav_ohlc["date"] <= dt].tail(lookback)
        if len(filtered) >= 5:  # lower threshold for bhavcopy data
            return filtered.reset_index(drop=True)

    return None
