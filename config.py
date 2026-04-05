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
NSE_EQUITY_OPTION_CHAIN_URL = NSE_BASE_URL + "/api/option-chain-equities?symbol={symbol}"
NSE_MARKET_TURNOVER_URL = NSE_BASE_URL + "/api/market-turnover"
NSE_ADVANCES_DECLINES_URL = NSE_BASE_URL + "/api/market-data-pre-open?key=ALL"
NSE_TOP_GAINERS_URL = NSE_BASE_URL + "/api/live-analysis-variations?index=gainers"
NSE_TOP_LOSERS_URL = NSE_BASE_URL + "/api/live-analysis-variations?index=losers"
NSE_LIVE_INDEX_URL = NSE_BASE_URL + "/api/equity-stockIndices?index={index}"
NSE_PRE_OPEN_URL = NSE_BASE_URL + "/api/market-data-pre-open?key={key}"

# ── NSE Archive endpoints (bhavcopy) ──────────────────────────────────────
NSE_FO_BHAVCOPY_URL = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES"
    "/{year}/{month}/fo{date}bhav.csv.zip"
)
NSE_CM_BHAVCOPY_URL = (
    "https://archives.nseindia.com/content/historical/EQUITIES"
    "/{year}/{month}/cm{date}bhav.csv.zip"
)

# ── Data persistence ──────────────────────────────────────────────────────
import pathlib as _pathlib
DATA_DIR = _pathlib.Path.home() / ".indian-options"
OI_DB_PATH = DATA_DIR / "oi_history.db"
BHAVCOPY_CACHE_DIR = DATA_DIR / "bhavcopy"

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

# ── Popular F&O stocks ────────────────────────────────────────────────────
FNO_STOCKS = {
    "RELIANCE": {"lot_size": 250, "strike_gap": 20},
    "TCS": {"lot_size": 175, "strike_gap": 50},
    "INFY": {"lot_size": 400, "strike_gap": 25},
    "HDFCBANK": {"lot_size": 550, "strike_gap": 25},
    "ICICIBANK": {"lot_size": 700, "strike_gap": 25},
    "SBIN": {"lot_size": 750, "strike_gap": 10},
    "BHARTIARTL": {"lot_size": 475, "strike_gap": 25},
    "ITC": {"lot_size": 1600, "strike_gap": 10},
    "TATAMOTORS": {"lot_size": 1400, "strike_gap": 10},
    "LT": {"lot_size": 150, "strike_gap": 50},
    "AXISBANK": {"lot_size": 625, "strike_gap": 25},
    "KOTAKBANK": {"lot_size": 400, "strike_gap": 25},
    "BAJFINANCE": {"lot_size": 125, "strike_gap": 100},
    "MARUTI": {"lot_size": 100, "strike_gap": 100},
    "TATASTEEL": {"lot_size": 5500, "strike_gap": 5},
}

# ── Backtester defaults ──────────────────────────────────────────────────
BACKTEST_DEFAULT_CAPITAL = 500_000  # INR
BACKTEST_DEFAULT_DTE_MIN = 7
BACKTEST_DEFAULT_DTE_MAX = 45
BACKTEST_SLIPPAGE_PCT = 0.5  # % of premium as estimated slippage
