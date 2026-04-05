"""
Market data functions – advances/declines, top movers,
stock option chains, and market breadth via NSE API.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config import (
    NSE_EQUITY_OPTION_CHAIN_URL,
    NSE_TOP_GAINERS_URL,
    NSE_TOP_LOSERS_URL,
    NSE_ADVANCES_DECLINES_URL,
    NSE_LIVE_INDEX_URL,
    NSE_MARKET_TURNOVER_URL,
    NSE_PRE_OPEN_URL,
    IST,
)
from services.nse_client import get_nse_client

logger = logging.getLogger(__name__)


def fetch_stock_option_chain(symbol: str) -> tuple[Optional[dict], datetime]:
    """
    Fetch option chain for an equity stock (e.g., RELIANCE, INFY).
    Returns the raw NSE JSON payload.
    """
    url = NSE_EQUITY_OPTION_CHAIN_URL.format(symbol=symbol)
    client = get_nse_client()
    data = client.get(url)
    if data and data.get("records"):
        return data, datetime.now(IST)
    return None, datetime.now(IST)


def _fetch_top_movers(url: str) -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch top gaining or losing stocks from NSE (shared logic)."""
    client = get_nse_client()
    data = client.get(url)
    if not data:
        return None, datetime.now(IST)
    try:
        items = data.get("NIFTY", data.get("data", []))
        if isinstance(items, list):
            df = pd.DataFrame(items)
            keep = [c for c in ["symbol", "lastPrice", "change", "pChange", "totalTradedVolume"] if c in df.columns]
            if keep:
                df = df[keep].head(15)
                df.columns = ["Symbol", "LTP", "Change", "Change %", "Volume"][:len(keep)]
                return df, datetime.now(IST)
    except Exception as exc:
        logger.warning("Top movers parse failed: %s", exc)
    return None, datetime.now(IST)


def fetch_top_gainers() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch top gaining stocks from NSE."""
    return _fetch_top_movers(NSE_TOP_GAINERS_URL)


def fetch_top_losers() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch top losing stocks from NSE."""
    return _fetch_top_movers(NSE_TOP_LOSERS_URL)


def fetch_advances_declines() -> tuple[Optional[dict], datetime]:
    """
    Fetch market breadth (advances vs declines) from NSE pre-open data.
    Returns dict: {advances, declines, unchanged, total}
    """
    client = get_nse_client()
    data = client.get(NSE_ADVANCES_DECLINES_URL)
    if not data:
        return None, datetime.now(IST)
    try:
        items = data.get("data", [])
        if not items:
            return None, datetime.now(IST)

        advances = 0
        declines = 0
        unchanged = 0
        for item in items:
            metadata = item.get("metadata", {})
            change = metadata.get("change", metadata.get("pChange", 0))
            try:
                change = float(change)
            except (ValueError, TypeError):
                continue
            if change > 0:
                advances += 1
            elif change < 0:
                declines += 1
            else:
                unchanged += 1

        return {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "total": advances + declines + unchanged,
            "ad_ratio": round(advances / declines, 2) if declines > 0 else advances,
        }, datetime.now(IST)
    except Exception as exc:
        logger.warning("Advances/declines parse failed: %s", exc)
        return None, datetime.now(IST)


def fetch_market_turnover() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch market turnover data from NSE."""
    client = get_nse_client()
    data = client.get(NSE_MARKET_TURNOVER_URL)
    if not data:
        return None, datetime.now(IST)
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            items = data.get("data", [data])
            df = pd.DataFrame(items if isinstance(items, list) else [items])
        else:
            return None, datetime.now(IST)
        return df, datetime.now(IST)
    except Exception as exc:
        logger.warning("Turnover parse failed: %s", exc)
        return None, datetime.now(IST)


def fetch_live_index(index_name: str = "NIFTY 50") -> tuple[Optional[dict], datetime]:
    """Fetch live index data with constituents."""
    url = NSE_LIVE_INDEX_URL.format(index=index_name)
    client = get_nse_client()
    data = client.get(url)
    if data:
        return data, datetime.now(IST)
    return None, datetime.now(IST)


def fetch_pre_open(key: str = "NIFTY") -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch pre-open market data."""
    url = NSE_PRE_OPEN_URL.format(key=key)
    client = get_nse_client()
    data = client.get(url)
    if not data:
        return None, datetime.now(IST)
    try:
        items = data.get("data", [])
        if not items:
            return None, datetime.now(IST)
        records = []
        for item in items:
            meta = item.get("metadata", {})
            records.append({
                "Symbol": meta.get("symbol", ""),
                "Open": meta.get("iep", 0),
                "Prev Close": meta.get("previousClose", 0),
                "Change": meta.get("change", 0),
                "Change %": meta.get("pChange", 0),
                "Final Qty": meta.get("finalQuantity", 0),
            })
        df = pd.DataFrame(records)
        df.sort_values("Change %", ascending=False, inplace=True)
        return df, datetime.now(IST)
    except Exception as exc:
        logger.warning("Pre-open parse failed: %s", exc)
        return None, datetime.now(IST)
