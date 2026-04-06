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
    NSE_OPTION_CHAIN_V3_URL,
    NSE_OPTION_CHAIN_CONTRACT_INFO_URL,
    NSE_TOP_GAINERS_URL,
    NSE_TOP_LOSERS_URL,
    NSE_ADVANCES_DECLINES_URL,
    NSE_LIVE_INDEX_URL,
    NSE_MARKET_TURNOVER_URL,
    NSE_PRE_OPEN_URL,
    NSE_ALL_INDICES_URL,
    IST,
)
from services.nse_client import get_nse_client

logger = logging.getLogger(__name__)


def fetch_stock_option_chain(symbol: str) -> tuple[Optional[dict], datetime]:
    """
    Fetch option chain for an equity stock (e.g., RELIANCE, INFY).
    Tries the v3 endpoint first, falls back to legacy.
    Returns the raw NSE JSON payload.
    """
    client = get_nse_client()

    # ── v3 path ────────────────────────────────────────────────────────
    info_url = NSE_OPTION_CHAIN_CONTRACT_INFO_URL.format(symbol=symbol)
    info = client.get(info_url)
    if info and isinstance(info.get("expiryDates"), list) and info["expiryDates"]:
        nearest = info["expiryDates"][0]
        url = NSE_OPTION_CHAIN_V3_URL.format(
            type="Equities", symbol=symbol, expiry=nearest,
        )
        data = client.get(url)
        if data and data.get("records"):
            return data, datetime.now(IST)

    # ── Legacy fallback ────────────────────────────────────────────────
    url = NSE_EQUITY_OPTION_CHAIN_URL.format(symbol=symbol)
    data = client.get(url)
    if data and data.get("records"):
        return data, datetime.now(IST)
    return None, datetime.now(IST)


def _fetch_index_constituents(
    index_name: str = "NIFTY 50",
) -> tuple[Optional[list[dict]], datetime]:
    """Fetch live constituent data for an index. Returns list of stock dicts."""
    url = NSE_LIVE_INDEX_URL.format(index=index_name)
    client = get_nse_client()
    data = client.get(url)
    if not data or "data" not in data:
        return None, datetime.now(IST)
    # First item is the index itself – skip it
    stocks = [
        s for s in data["data"]
        if s.get("symbol") != index_name and s.get("symbol", "").upper() != index_name.replace(" ", "")
    ]
    return stocks, datetime.now(IST)


def _stocks_to_movers_df(stocks: list[dict], top_n: int = 15) -> pd.DataFrame:
    """Convert raw stock dicts to a clean DataFrame."""
    rows = []
    for s in stocks[:top_n]:
        rows.append({
            "Symbol": s.get("symbol", ""),
            "LTP": s.get("lastPrice", 0),
            "Change": round(float(s.get("change", 0)), 2),
            "Change %": round(float(s.get("pChange", 0)), 2),
            "Volume": int(s.get("totalTradedVolume", 0)),
        })
    return pd.DataFrame(rows)


def fetch_top_gainers() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch top gaining stocks from NIFTY 50 constituents."""
    stocks, ts = _fetch_index_constituents("NIFTY 50")
    if not stocks:
        # Fallback: try the old gainers-specific endpoint
        return _fetch_top_movers_legacy(NSE_TOP_GAINERS_URL)
    gainers = sorted(stocks, key=lambda s: float(s.get("pChange", 0)), reverse=True)
    gainers = [s for s in gainers if float(s.get("pChange", 0)) > 0]
    if not gainers:
        return None, ts
    return _stocks_to_movers_df(gainers), ts


def fetch_top_losers() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch top losing stocks from NIFTY 50 constituents."""
    stocks, ts = _fetch_index_constituents("NIFTY 50")
    if not stocks:
        return None, datetime.now(IST)
    losers = sorted(stocks, key=lambda s: float(s.get("pChange", 0)))
    losers = [s for s in losers if float(s.get("pChange", 0)) < 0]
    if not losers:
        return None, ts
    return _stocks_to_movers_df(losers), ts


def _fetch_top_movers_legacy(url: str) -> tuple[Optional[pd.DataFrame], datetime]:
    """Legacy fallback: parse old-format gainers/losers response."""
    client = get_nse_client()
    data = client.get(url)
    if not data:
        return None, datetime.now(IST)
    try:
        # New NSE format: {"NIFTY": {"data": [...]}, "allSec": {"data": [...]}}
        items = None
        for key in ("NIFTY", "allSec"):
            sub = data.get(key)
            if isinstance(sub, dict):
                items = sub.get("data", [])
                if items:
                    break
        # Old format: {"data": [...]}
        if not items:
            items = data.get("data", [])
        if not isinstance(items, list) or not items:
            return None, datetime.now(IST)

        df = pd.DataFrame(items)
        # Handle both old field names (lastPrice, pChange) and new (ltp, net_price)
        col_map = {}
        if "symbol" in df.columns:
            col_map["symbol"] = "Symbol"
        if "ltp" in df.columns:
            col_map["ltp"] = "LTP"
        elif "lastPrice" in df.columns:
            col_map["lastPrice"] = "LTP"
        if "net_price" in df.columns:
            col_map["net_price"] = "Change %"
        elif "pChange" in df.columns:
            col_map["pChange"] = "Change %"
        if "trade_quantity" in df.columns:
            col_map["trade_quantity"] = "Volume"
        elif "totalTradedVolume" in df.columns:
            col_map["totalTradedVolume"] = "Volume"

        if not col_map:
            return None, datetime.now(IST)

        df = df[list(col_map.keys())].head(15)
        df.rename(columns=col_map, inplace=True)
        return df, datetime.now(IST)
    except Exception as exc:
        logger.warning("Top movers legacy parse failed: %s", exc)
    return None, datetime.now(IST)


def fetch_sector_performance() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch sector-wise index performance from NSE allIndices."""
    client = get_nse_client()
    data = client.get(NSE_ALL_INDICES_URL)
    if not data:
        return None, datetime.now(IST)
    try:
        items = data.get("data", [])
        # Pick key sectoral indices
        sector_keywords = {
            "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY FMCG",
            "NIFTY AUTO", "NIFTY METAL", "NIFTY REALTY", "NIFTY ENERGY",
            "NIFTY INFRA", "NIFTY PSE", "NIFTY MEDIA", "NIFTY FIN SERVICE",
            "NIFTY PRIVATE BANK", "NIFTY PSU BANK", "NIFTY HEALTHCARE",
            "NIFTY CONSUMER DURABLES", "NIFTY OIL AND GAS",
        }
        rows = []
        for item in items:
            idx_name = (item.get("index") or "").upper()
            if idx_name in sector_keywords:
                rows.append({
                    "Sector": item.get("index", ""),
                    "Last": round(float(item.get("last", 0)), 1),
                    "Change": round(float(item.get("percentChange", 0)), 2),
                    "Open": round(float(item.get("open", 0)), 1),
                    "High": round(float(item.get("high", 0)), 1),
                    "Low": round(float(item.get("low", 0)), 1),
                })
        if not rows:
            return None, datetime.now(IST)
        df = pd.DataFrame(rows)
        df.sort_values("Change", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, datetime.now(IST)
    except Exception as exc:
        logger.warning("Sector performance parse failed: %s", exc)
        return None, datetime.now(IST)


def fetch_advances_declines() -> tuple[Optional[dict], datetime]:
    """
    Fetch market breadth (advances vs declines) from NSE pre-open data.
    Returns dict: {advances, declines, unchanged, total, ad_ratio,
                   totalTradedValue, totalMarketCap}
    """
    client = get_nse_client()
    data = client.get(NSE_ADVANCES_DECLINES_URL)
    if not data:
        return None, datetime.now(IST)
    try:
        # NSE now provides top-level advance/decline counts directly
        adv = data.get("advances")
        dec = data.get("declines")
        unc = data.get("unchanged")

        if adv is not None and dec is not None:
            adv, dec = int(adv), int(dec)
            unc = int(unc) if unc is not None else 0
        else:
            # Fallback: count manually from data items
            items = data.get("data", [])
            if not items:
                return None, datetime.now(IST)
            adv = dec = unc = 0
            for item in items:
                metadata = item.get("metadata", {})
                change = metadata.get("change", metadata.get("pChange", 0))
                try:
                    change = float(change)
                except (ValueError, TypeError):
                    continue
                if change > 0:
                    adv += 1
                elif change < 0:
                    dec += 1
                else:
                    unc += 1

        total = adv + dec + unc
        return {
            "advances": adv,
            "declines": dec,
            "unchanged": unc,
            "total": total,
            "ad_ratio": round(adv / dec, 2) if dec > 0 else float(adv),
            "totalTradedValue": data.get("totalTradedValue"),
            "totalMarketCap": data.get("totalmarketcap"),
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
