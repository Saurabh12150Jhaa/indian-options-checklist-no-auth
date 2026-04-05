"""Shared utilities for backtester strategy modules."""

from datetime import date
from typing import Optional

import pandas as pd


def get_recent_ohlc(engine, dt: date, lookback: int = 50) -> Optional[pd.DataFrame]:
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
