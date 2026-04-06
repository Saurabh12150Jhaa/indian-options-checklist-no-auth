"""
Data fetching module – zero-auth, free public APIs only.

Sources:
  - yfinance : Global indices, India VIX, index spot prices, historical candles
  - NSE API  : Option chain (OI, IV, greeks), FII/DII activity, VIX fallback
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    INDEX_CONFIG,
    INDIA_VIX_TICKER,
    GLOBAL_TICKERS,
    NSE_OPTION_CHAIN_URL,
    NSE_OPTION_CHAIN_V3_URL,
    NSE_OPTION_CHAIN_CONTRACT_INFO_URL,
    NSE_FII_DII_URL,
    NSE_ALL_INDICES_URL,
    IST,
)
from services.nse_client import get_nse_client

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  yfinance helpers
# ══════════════════════════════════════════════════════════════════════════


def fetch_global_cues() -> tuple[pd.DataFrame, datetime]:
    """Fetch overnight/pre-market global index data via yfinance."""
    records = []
    for name, ticker in GLOBAL_TICKERS.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if hist.empty:
                continue
            prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else hist["Close"].iloc[-1]
            last_close = hist["Close"].iloc[-1]
            pct = ((last_close - prev_close) / prev_close) * 100
            records.append({
                "Index": name,
                "Last Close": round(float(last_close), 2),
                "Prev Close": round(float(prev_close), 2),
                "Change %": round(float(pct), 2),
            })
        except Exception as exc:
            logger.warning("yfinance %s failed: %s", name, exc)
    return pd.DataFrame(records), datetime.now(IST)


def fetch_india_vix() -> tuple[Optional[float], datetime]:
    """Fetch India VIX latest value. Tries yfinance first, then NSE allIndices."""
    # Primary: yfinance
    try:
        tk = yf.Ticker(INDIA_VIX_TICKER)
        hist = tk.history(period="5d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2), datetime.now(IST)
    except Exception as exc:
        logger.warning("India VIX (yfinance) failed: %s", exc)

    # Fallback: NSE allIndices
    try:
        client = get_nse_client()
        data = client.get(NSE_ALL_INDICES_URL)
        if data:
            for item in data.get("data", []):
                if "VIX" in (item.get("index") or "").upper():
                    val = item.get("last")
                    if val is not None:
                        return round(float(val), 2), datetime.now(IST)
    except Exception as exc:
        logger.warning("India VIX (NSE) failed: %s", exc)

    return None, datetime.now(IST)


def fetch_spot_data(index_name: str) -> tuple[Optional[dict], datetime]:
    """Fetch current spot OHLC for an index via yfinance."""
    cfg = INDEX_CONFIG.get(index_name)
    if not cfg:
        return None, datetime.now(IST)
    try:
        tk = yf.Ticker(cfg["yf_ticker"])
        hist = tk.history(period="5d")
        if hist.empty:
            return None, datetime.now(IST)
        today = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else today
        return {
            "ltp": round(float(today["Close"]), 2),
            "open": round(float(today["Open"]), 2),
            "high": round(float(today["High"]), 2),
            "low": round(float(today["Low"]), 2),
            "prev_close": round(float(prev["Close"]), 2),
            "volume": int(today["Volume"]),
        }, datetime.now(IST)
    except Exception as exc:
        logger.warning("Spot data fetch failed for %s: %s", index_name, exc)
        return None, datetime.now(IST)


def fetch_historical(index_name: str, period: str = "1y") -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch daily historical candles for technical analysis."""
    cfg = INDEX_CONFIG.get(index_name)
    if not cfg:
        return None, datetime.now(IST)
    try:
        tk = yf.Ticker(cfg["yf_ticker"])
        df = tk.history(period=period)
        if df.empty:
            return None, datetime.now(IST)
        df.columns = [c.lower() for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep], datetime.now(IST)
    except Exception as exc:
        logger.warning("Historical fetch failed for %s: %s", index_name, exc)
        return None, datetime.now(IST)


def fetch_intraday(index_name: str, interval: str = "15m") -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch intraday candles (last 5 days, yfinance limit)."""
    cfg = INDEX_CONFIG.get(index_name)
    if not cfg:
        return None, datetime.now(IST)
    try:
        tk = yf.Ticker(cfg["yf_ticker"])
        df = tk.history(period="5d", interval=interval)
        if df.empty:
            return None, datetime.now(IST)
        df.columns = [c.lower() for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep], datetime.now(IST)
    except Exception as exc:
        logger.warning("Intraday fetch failed for %s: %s", index_name, exc)
        return None, datetime.now(IST)


# ══════════════════════════════════════════════════════════════════════════
#  NSE API helpers
# ══════════════════════════════════════════════════════════════════════════


def fetch_nse_expiry_dates(index_name: str) -> list[str]:
    """
    Fetch available expiry dates for an index from NSE contract-info endpoint.
    Returns a list of expiry date strings (e.g. ['07-Apr-2026', ...]) or [].
    """
    cfg = INDEX_CONFIG.get(index_name)
    if not cfg:
        return []
    url = NSE_OPTION_CHAIN_CONTRACT_INFO_URL.format(symbol=cfg["nse_symbol"])
    client = get_nse_client()
    data = client.get(url)
    if data and isinstance(data.get("expiryDates"), list):
        return data["expiryDates"]
    return []


def fetch_nse_option_chain(
    index_name: str,
    expiry: Optional[str] = None,
) -> tuple[Optional[dict], datetime]:
    """
    Fetch option chain from NSE for the given index.

    Tries the v3 endpoint first (``/api/option-chain-v3``), which requires
    an explicit *expiry* date.  If *expiry* is not supplied we first ask the
    contract-info endpoint for available dates and pick the nearest one.

    Falls back to the legacy ``/api/option-chain-indices`` endpoint when v3
    fails (e.g. if NSE rolls things back).

    Returns the raw NSE JSON payload:
      { "records": { "data": [...], "expiryDates": [...],
                     "strikePrices": [...], "underlyingValue": ... },
        "filtered": { "data": [...], "CE": {...}, "PE": {...} } }

    Note: NSE returns {} on non-trading days / outside market hours.
    """
    cfg = INDEX_CONFIG.get(index_name)
    if not cfg:
        return None, datetime.now(IST)

    client = get_nse_client()

    # ── v3 path (preferred) ────────────────────────────────────────────
    target_expiry = expiry
    if target_expiry is None:
        expiry_list = fetch_nse_expiry_dates(index_name)
        if expiry_list:
            target_expiry = expiry_list[0]  # nearest expiry

    if target_expiry:
        url = NSE_OPTION_CHAIN_V3_URL.format(
            type="Indices",
            symbol=cfg["nse_symbol"],
            expiry=target_expiry,
        )
        data = client.get(url)
        if data and data.get("records"):
            return data, datetime.now(IST)
        logger.debug("v3 option-chain returned empty for %s expiry %s", index_name, target_expiry)

    # ── Legacy fallback ────────────────────────────────────────────────
    url = NSE_OPTION_CHAIN_URL.format(symbol=cfg["nse_symbol"])
    data = client.get(url)
    if data and data.get("records"):
        return data, datetime.now(IST)

    return None, datetime.now(IST)


def fetch_fii_dii() -> tuple[Optional[pd.DataFrame], datetime]:
    """Fetch FII/DII cash-market activity from NSE (fiidiiTradeReact)."""
    client = get_nse_client()
    data = client.get(NSE_FII_DII_URL)
    if not data:
        return None, datetime.now(IST)
    try:
        # fiidiiTradeReact returns a list of category dicts
        items = data if isinstance(data, list) else data.get("data", data)
        if not isinstance(items, list):
            return None, datetime.now(IST)

        records = []
        for entry in items:
            records.append({
                "Category": entry.get("category", ""),
                "Date": entry.get("date", ""),
                "Buy Value (Cr)": entry.get("buyValue", ""),
                "Sell Value (Cr)": entry.get("sellValue", ""),
                "Net Value (Cr)": entry.get("netValue", ""),
            })
        return pd.DataFrame(records), datetime.now(IST)
    except Exception as exc:
        logger.warning("FII/DII parse failed: %s", exc)
        return None, datetime.now(IST)
