"""
Configuration for the Indian Options Pre-Trade Checklist.
No API keys required – uses only free public data sources.
"""

import os
import ssl
from datetime import time as dtime
from zoneinfo import ZoneInfo

# ── SSL / Zscaler proxy fix ───────────────────────────────────────────────
# Detect the system OpenSSL cert bundle (which includes any corporate CA
# roots injected by Zscaler / other TLS-inspecting proxies) and make sure
# *every* HTTP library in the process honours it.
_SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE") or os.environ.get(
    "REQUESTS_CA_BUNDLE"
)
if not _SSL_CERT_FILE:
    _default = ssl.get_default_verify_paths().cafile
    if _default and os.path.isfile(_default):
        _SSL_CERT_FILE = _default
if _SSL_CERT_FILE:
    os.environ.setdefault("SSL_CERT_FILE", _SSL_CERT_FILE)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _SSL_CERT_FILE)
    os.environ.setdefault("CURL_CA_BUNDLE", _SSL_CERT_FILE)

# ── Timezone ───────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")

# ── Market hours (IST) ────────────────────────────────────────────────────
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
PRE_MARKET_OPEN = dtime(9, 0)

# ── NSE endpoints ─────────────────────────────────────────────────────────
NSE_BASE_URL = "https://www.nseindia.com"
NSE_OPTION_CHAIN_URL = NSE_BASE_URL + "/api/option-chain-indices?symbol={symbol}"
NSE_FII_DII_URL = NSE_BASE_URL + "/api/fiidiiTradeReact"
NSE_ALL_INDICES_URL = NSE_BASE_URL + "/api/allIndices"
NSE_MARKET_STATUS_URL = NSE_BASE_URL + "/api/marketStatus"

# ── Index definitions ─────────────────────────────────────────────────────
INDEX_CONFIG = {
    "NIFTY": {
        "nse_symbol": "NIFTY",
        "yf_ticker": "^NSEI",
        "display_name": "NIFTY 50",
        "lot_size": 25,
        "strike_gap": 50,
    },
    "BANKNIFTY": {
        "nse_symbol": "BANKNIFTY",
        "yf_ticker": "^NSEBANK",
        "display_name": "BANK NIFTY",
        "lot_size": 15,
        "strike_gap": 100,
    },
    "FINNIFTY": {
        "nse_symbol": "NIFTY FIN SERVICE",
        "yf_ticker": "NIFTY_FIN_SERVICE.NS",
        "display_name": "FIN NIFTY",
        "lot_size": 25,
        "strike_gap": 50,
    },
}

# ── yfinance tickers ──────────────────────────────────────────────────────
INDIA_VIX_TICKER = "^INDIAVIX"

GLOBAL_TICKERS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "GIFT Nifty": "0QF5.L",
}

# ── Technical analysis parameters ─────────────────────────────────────────
EMA_PERIODS = [9, 20, 50, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
