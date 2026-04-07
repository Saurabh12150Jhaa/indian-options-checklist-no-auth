# Indian Index Options Pre-Trade Checklist

A comprehensive Streamlit dashboard for Indian options trading analysis. Gathers and visualizes global cues, technical indicators, options chain data, market breadth, sentiment-tagged news, and includes a full strategy backtester -- all using **free public APIs** with **zero authentication**.

---

## Screenshots

<details>
<summary><strong>Pre-Trade Checklist</strong> -- Global cues, VIX, FII/DII, technicals, 10-point weighted checklist</summary>

![Pre-Trade Checklist](screenshots/tab1_checklist.png)
</details>

<details>
<summary><strong>OI Deep Dive</strong> -- Multi-expiry OI comparison, PCR timeline, ITM/OTM analysis</summary>

![OI Deep Dive](screenshots/tab2_oi_deep_dive.png)
</details>

<details>
<summary><strong>Stock Options</strong> -- Individual F&O stock option chains with all strikes</summary>

![Stock Options](screenshots/tab3_stock_options.png)
</details>

<details>
<summary><strong>Market Breadth</strong> -- Advances/declines, A/D ratio, top gainers & losers</summary>

![Market Breadth](screenshots/tab4_market_breadth.png)
</details>

<details>
<summary><strong>Strategy Backtester</strong> -- 15+ strategies, custom builder, real P&L from NSE bhavcopy data</summary>

![Strategy Backtester](screenshots/tab5_backtester.png)
</details>

<details>
<summary><strong>Market News & Impact</strong> -- Sentiment-tagged news with category and sentiment filters</summary>

![Market News](screenshots/tab6_market_news.png)
</details>

---

## Data Sources

All data is fetched from free, publicly available APIs. No broker account or API key is needed.

| Data | Source | Update Frequency |
|---|---|---|
| Global Indices (S&P 500, NASDAQ, DAX, etc.) | yfinance | Overnight closes |
| GIFT Nifty | yfinance | Pre-market indicator |
| India VIX | yfinance / NSE | ~120s cache TTL |
| Spot Price & Historical Candles | yfinance | NIFTY, BANKNIFTY, FINNIFTY |
| Option Chain (OI, IV, Greeks) | NSE India API | ~90s cache TTL |
| FII / DII Activity | NSE India API | ~300s cache TTL |
| Market Breadth, Top Movers, Pre-Open | NSE India API | Real-time during market |
| F&O Bhavcopy (backtesting) | NSE Archives | Auto-downloaded per backtest |
| Market News & Sentiment | RSS (ET, LiveMint, Business Standard, Google News) | ~120s cache TTL |

---

## Features

### Tab 1: Pre-Trade Checklist

**Section 1 -- Global Cues & Macro**
- Global indices with overnight change %
- India VIX with threshold-based signalling
- FII/DII net buy/sell activity

**Section 2 -- Spot & Technical Analysis**
- Spot price with OHLC and % change
- Moving averages: EMA 9, 20, 50, 200 with above/below positioning
- Momentum: RSI (14), MACD with histogram
- Classic pivot points (PP, R1-R3, S1-S3)
- Fibonacci retracement (auto swing high/low detection)
- Interactive candlestick chart (intraday 5m/15m/30m/60m or daily)

**Section 3 -- Options Chain Analysis**
- PCR (Put-Call Ratio) by OI and volume
- Max pain strike calculation
- Highest call OI (resistance) and highest put OI (support)
- OI buildup analysis (where new OI is being created)
- ATM straddle price with expected move % and range
- IV smile chart across strikes
- OI distribution and OI change bar charts

**Section 4 -- Pre-Trade Checklist Summary**
- 10-point weighted checklist: each indicator scored BULLISH / BEARISH / NEUTRAL
- Overall bias score (-100% to +100%) with colour-coded gauge
- Signal summary bar showing bullish/neutral/bearish distribution
- Graceful degradation: if NSE is unavailable, yfinance sections still work

### Tab 2: OI Deep Dive
- Multi-expiry OI comparison (up to 4 nearest expiries)
- Aggregate OI timeline with PCR over time (intraday tracking via SQLite)
- ITM / OTM analysis with call and put OI breakdowns
- Data flows from Tab 1 via `st.session_state`

### Tab 3: Stock Options
- Individual F&O stock option chain analysis
- All NSE F&O stocks (RELIANCE, TCS, INFY, HDFCBANK, etc.)
- Key metrics: PCR, max pain, lot size, ATM straddle, IV
- OI distribution and OI change charts per stock

### Tab 4: Market Breadth
- Advances vs declines with count metrics and A/D ratio
- Donut chart visualization
- Top gainers and top losers tables
- Pre-open market data (available 9:00-9:15 IST)

### Tab 5: Strategy Backtester
- **15+ pre-built strategies** across categories:
  - *Neutral*: Short Straddle, Short Strangle, Iron Butterfly, Iron Condor
  - *Bullish*: Long Call, Bull Call Spread, Bull Put Spread
  - *Bearish*: Long Put, Bear Put Spread, Bear Call Spread
  - *Volatility*: Long Straddle, Long Strangle
  - *Smart Money (5)*: OB Entry, Liquidity Sweep, FVG Fill, CHoCH Reversal
  - *Price Action (5)*: Pin Bar, Engulfing, Inside Bar Breakout, S/R Breakout, EMA Pullback
- **Custom strategy builder** with multi-leg support and condition registry
- **Auto-download** of NSE bhavcopy data (new + old NSE archive formats)
- **Data management**: download, cleanup, and cache stats
- Configurable: symbol, date range, initial capital, min DTE, stop loss %, position sizing

### Tab 6: Market News & Impact
- News aggregation from Economic Times, LiveMint, Business Standard, Google News RSS
- Sentiment analysis (Positive / Negative / Neutral) per article
- Filter by category (Options/F&O, Market Outlook, Macro/Policy, Sector/Stock, Technical)
- Sentiment distribution donut chart
- LIVE badge for articles published within the last 2 hours

### UX
- **Market phase indicator**: PRE-MARKET / MARKET OPEN / POST-MARKET badge
- **Auto-refresh**: configurable JS-based timer during market hours (30s-5m)
- **Data freshness timestamps**: each section shows when data was last fetched
- **Robust NSE client**: automatic cookie management with TLS fingerprint impersonation (curl_cffi), exponential backoff with jitter, session auto-refresh
- **Dark theme**: custom CSS with colour-coded signals, gradient bias boxes

---

## Quick Start

### Prerequisites
- Python 3.10 -- 3.13 (recommended: 3.12)

### Install & Run

```bash
git clone https://github.com/Saurabh12150Jhaa/indian-options-checklist-no-auth.git
cd indian-options-checklist-no-auth

# Create virtual environment (uv recommended, pip works too)
uv venv --python 3.12 .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
uv pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at `http://localhost:8501`. No credentials needed -- select an index and start analysing.

### Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app.py`
4. In **Advanced Settings**, select Python **3.12** (do **not** use 3.14)
5. Deploy -- no secrets or env vars required

---

## Project Structure

```
indian-options-checklist-no-auth/
│
├── app.py                        # Slim orchestrator (108 lines) -- sidebar, tabs, wiring
├── config.py                     # Index definitions, NSE endpoints, market hours, SSL fix
│
├── core/                         # Shared foundations
│   ├── models.py                 # BULLISH/BEARISH/NEUTRAL constants
│   ├── exceptions.py             # AppError -> DataFetchError, NSEConnectionError, etc.
│   └── protocols.py              # Protocol ABCs (MarketDataProvider, OptionChainProvider, ...)
│
├── analysis/                     # Pure analysis logic (no I/O)
│   ├── technical.py              # EMA, RSI, MACD, pivots, fibonacci, run_technical_analysis()
│   ├── options.py                # parse_nse_option_chain(), PCR, max pain, IV, straddle
│   └── signals.py                # generate_checklist(), compute_overall_bias(), market_phase()
│
├── services/                     # Data layer (all I/O lives here)
│   ├── nse_client.py             # NSE session manager (cookies, TLS impersonation, backoff)
│   ├── data_fetcher.py           # yfinance + NSE API fetchers (spot, historical, option chain)
│   ├── market_data.py            # Market breadth, top movers, stock option chains, pre-open
│   ├── news_service.py           # RSS news aggregation with sentiment classification
│   ├── oi_tracker.py             # SQLite-backed OI snapshot persistence and timeline queries
│   └── bhavcopy/                 # Bhavcopy data management
│       ├── cache.py              # Shared access-log, date helpers, filename generators
│       ├── downloader.py         # Auto-download from NSE archives (new + old format)
│       └── cleaner.py            # Cleanup unused files, cache stats
│
├── ui/                           # Streamlit UI layer
│   ├── components.py             # Shared CSS injection, badges, HTML builders
│   ├── cache.py                  # @st.cache_data wrappers for all data fetchers
│   ├── tab_checklist.py          # Tab 1: Pre-Trade Checklist (4 sections)
│   ├── tab_oi_deep_dive.py       # Tab 2: OI Deep Dive
│   ├── tab_stock_options.py      # Tab 3: Stock Options
│   ├── tab_market_breadth.py     # Tab 4: Market Breadth
│   ├── tab_backtester.py         # Tab 5: Strategy Backtester
│   └── tab_news.py               # Tab 6: Market News
│
├── backtester/                   # Strategy backtesting engine
│   ├── engine.py                 # Core backtesting loop and P&L calculation
│   ├── strategies.py             # 15+ pre-built option strategies
│   ├── smart_money.py            # OI-based and SMC strategies (5)
│   ├── price_action.py           # Price action strategies (5)
│   ├── custom_strategy.py        # Custom multi-leg strategy builder with condition registry
│   ├── data_adapter.py           # Bhavcopy CSV format adapter (old + new NSE formats)
│   └── utils.py                  # Shared helpers (get_recent_ohlc)
│
├── *.py (root-level facades)     # Thin re-exports from services/ for backward compatibility
│   ├── nse_client.py             #   -> services.nse_client
│   ├── data_fetcher.py           #   -> services.data_fetcher
│   ├── market_data.py            #   -> services.market_data
│   ├── news_fetcher.py           #   -> services.news_service
│   ├── oi_tracker.py             #   -> services.oi_tracker
│   ├── data_downloader.py        #   -> services.bhavcopy.downloader + cache
│   └── data_cleaner.py           #   -> services.bhavcopy.cleaner
│
├── requirements.txt              # 10 dependencies (all Python 3.14 compatible)
├── .streamlit/config.toml        # Dark theme, headless server config
├── .devcontainer/                # VS Code Dev Container config
├── bhavcopy/                     # Downloaded bhavcopy files (auto-managed, gitignored)
└── screenshots/                  # App screenshots
```

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                  Streamlit Dashboard                     │
                    │                     (app.py)                             │
                    │                                                          │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
                    │  │Checklist │ │ OI Deep  │ │  Stock   │ │  Market   │  │
                    │  │  Tab 1   │ │ Dive (2) │ │ Opts (3) │ │Breadth(4) │  │
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │
                    │  ┌────┴─────┐ ┌────┴─────┐                              │
                    │  │Backtester│ │  News    │     ui/components.py          │
                    │  │  Tab 5   │ │  Tab 6   │     ui/cache.py              │
                    │  └────┬─────┘ └────┬─────┘                              │
                    └───────┼────────────┼────────────────────────────────────┘
                            │            │
              ┌─────────────┴────────────┴─────────────┐
              │            analysis/                     │
              │  technical.py  options.py  signals.py    │
              │  (EMA, RSI,    (PCR, max   (checklist,  │
              │   MACD, pivots) pain, IV)   bias score)  │
              └─────────────────┬───────────────────────┘
                                │
              ┌─────────────────┴───────────────────────┐
              │             services/                     │
              │                                          │
              │  nse_client.py ──── NSE India API        │
              │  data_fetcher.py ── yfinance + NSE       │
              │  market_data.py ─── Breadth, Movers      │
              │  news_service.py ── RSS Feeds             │
              │  oi_tracker.py ──── SQLite persistence    │
              │  bhavcopy/ ──────── NSE Archive downloads │
              └──────────────────────────────────────────┘
                                │
              ┌─────────────────┴───────────────────────┐
              │           External Data Sources           │
              │                                          │
              │  yfinance    NSE India    RSS Feeds       │
              │  (Yahoo)     API          (ET, LiveMint,  │
              │              + Archives    Business Std,   │
              │                            Google News)    │
              └──────────────────────────────────────────┘
```

The codebase follows a layered architecture:

- **`ui/`** -- presentation layer. Each tab is an independent module with a `render()` function. Shared CSS and HTML helpers live in `components.py`. Streamlit caching wrappers live in `cache.py`.
- **`analysis/`** -- pure business logic. No I/O, no Streamlit imports. Takes DataFrames in, returns results out.
- **`services/`** -- data access layer. All HTTP calls, SQLite queries, and file I/O happen here. Each service is independently testable.
- **`core/`** -- shared constants, exception hierarchy, and protocol definitions.
- **`backtester/`** -- self-contained backtesting engine with its own strategy registry and data adapter.

Root-level `.py` files (e.g., `data_fetcher.py`, `nse_client.py`) are thin facades that re-export from `services/` for backward compatibility.

---

## Backtester: Bhavcopy Data

The backtester uses NSE F&O bhavcopy CSV files for historical options data.

### Data Sources (priority order)
1. **NSE New Format** (`nsearchives.nseindia.com`) -- 2024 onwards
2. **NSE Old Format** (`archives.nseindia.com`) -- up to July 2024
3. **jugaad-data** package (if installed) -- Python package fallback

### How It Works
- When you run a backtest, missing bhavcopy files are **automatically downloaded** for each trading day in the date range
- Downloaded files are cached in `bhavcopy/` and tracked via `bhavcopy_access.json`
- Files unused for 30+ days are automatically cleaned up on app startup

---

## Dependencies

```
streamlit>=1.30.0          # Dashboard framework
streamlit-autorefresh>=1.0.1  # JS-based auto-refresh timer
pandas>=2.0.0              # DataFrames
yfinance>=0.2.36           # Yahoo Finance API
requests>=2.31.0           # HTTP client (fallback)
plotly>=5.18.0             # Interactive charts
numpy>=1.24.0              # Numerical computing
feedparser>=6.0.0          # RSS feed parsing
mibian>=0.1.3              # Black-Scholes options pricing
openpyxl>=3.1.0            # Excel file support
```

Optional: `curl_cffi` for TLS fingerprint impersonation (bypasses NSE Akamai bot protection). Falls back to standard `requests` if unavailable.

---

## Deployed link [indian-options-checklist-no-app](https://indian-options-checklist-no-app-cxkgahk3s6stuqbfoxcjjt.streamlit.app/)


## Usage Notes

- **Best during market hours** (9:15 AM -- 3:30 PM IST) for real-time NSE data
- **NSE rate limits**: the app caches aggressively (60-300s TTL) to avoid blocks
- **If NSE blocks**: click "Refresh Now" after a few seconds; the client auto-refreshes cookies
- **Graceful degradation**: if NSE is down, yfinance-sourced sections still work normally
- **Pre-open data**: available only during 9:00-9:15 IST window
- **Backtester date ranges**: 2024-present (new format) or pre-July 2024 (old format)
- **Corporate proxies**: the app auto-detects system SSL certificates (Zscaler-compatible)

---

## License

This tool is for **informational and educational purposes only**. It does not constitute financial advice. Always do your own analysis before making trading decisions.
