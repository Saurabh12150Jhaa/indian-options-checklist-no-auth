# Indian Index Options Pre-Trade Checklist (No Auth)

A Streamlit-based web application that automatically gathers and visualizes global cues, technical indicators, options chain data, and market breadth using **free public APIs** — no broker account or API keys required.

## Data Sources (Free, Zero Auth)

| Data | Source | Notes |
|---|---|---|
| Global Indices (S&P 500, NASDAQ, etc.) | yfinance | Overnight closes |
| GIFT Nifty | yfinance | Pre-market indicator |
| India VIX | yfinance | Option premium volatility |
| Spot Price & Historical Candles | yfinance | NIFTY, BANKNIFTY, FINNIFTY |
| Option Chain (OI, IV, Greeks) | NSE India API | Full chain with all strikes |
| FII / DII Activity | NSE India API | Cash market activity |

## Features

### 1. Global Cues & Macro
- Global indices with overnight change %
- India VIX with threshold-based signalling
- FII/DII net buy/sell activity

### 2. Technical Analysis
- Spot Price with OHLC and % change
- Moving Averages: 9, 20, 50, 200 EMA with above/below positioning
- Momentum: RSI (14), MACD with histogram
- Classic Pivot Points (PP, R1-R3, S1-S3)
- Fibonacci Retracement (auto swing high/low)
- Interactive intraday candlestick chart (5m / 15m / 30m / 60m)

### 3. Options Chain Analysis (via NSE)
- **PCR** (Put-Call Ratio) by OI and Volume
- **Max Pain** strike calculation
- **Highest Call OI** (Resistance) & **Highest Put OI** (Support)
- **OI Buildup** analysis (where new OI is being created)
- **ATM Straddle** price with expected move % and range
- **IV Smile** chart across strikes
- **OI Distribution** & OI Change bar charts

### 4. Pre-Trade Checklist Summary
- 10-point weighted checklist: each indicator scored BULLISH / BEARISH / NEUTRAL
- Overall bias score (-100% to +100%) with color-coded gauge
- Graceful degradation: if NSE is unavailable, yfinance data still shows

### 5. UX Enhancements
- **Market Phase Indicator**: PRE-MARKET / MARKET OPEN / POST-MARKET badge
- **Data Freshness Timestamps**: each section shows when data was last fetched
- **No Authentication Required**: just run and use
- **Robust NSE Client**: automatic cookie management, exponential backoff, session refresh

## Setup

### Prerequisites
- Python 3.10 - 3.13 (recommended: 3.12)

### Installation

```bash
cd indian-options-checklist-no-auth

# Create virtual environment (using uv — recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. No credentials needed — select an index and start analyzing.

## Project Structure

```
indian-options-checklist-no-auth/
├── app.py              # Streamlit dashboard (UI + layout)
├── config.py           # Index definitions, NSE endpoints, TA parameters
├── nse_client.py       # Robust NSE session manager (cookies, retries, backoff)
├── data_fetcher.py     # yfinance + NSE data fetching functions
├── analysis.py         # Technical indicators, options analytics, signal engine
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  yfinance    │    │  NSE India   │    │  Streamlit   │
│  (Global,    │───>│  (Option     │───>│  Dashboard   │
│   Spot, VIX) │    │   Chain,     │    │  (app.py)    │
└─────────────┘    │   FII/DII)   │    └──────┬───────┘
                   └──────┬───────┘           │
                          │                   v
                   ┌──────v───────┐    ┌──────────────┐
                   │ nse_client.py│    │ analysis.py  │
                   │ (Session,    │    │ (TA, Options,│
                   │  Cookies,    │    │  Signals)    │
                   │  Backoff)    │    └──────────────┘
                   └──────────────┘
```

## Usage Notes

- **Best during market hours** (9:15 AM – 3:30 PM IST) for real-time NSE data
- **NSE rate limits**: The app caches aggressively (60-300s TTL) to avoid blocks
- **If NSE blocks**: Click "Refresh All Data" after a few seconds; the client auto-refreshes cookies
- **Graceful degradation**: If NSE is down, yfinance sections still work normally

## Disclaimer

This tool is for **informational and educational purposes only**. It does not constitute financial advice. Always do your own analysis before making trading decisions.
