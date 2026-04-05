"""
Indian Index Options Pre-Trade Checklist – Streamlit Dashboard
==============================================================
Zero-auth. Uses NSE India API + yfinance only.
Run with:  streamlit run app.py

This is the slim orchestrator that wires together the UI tab modules.
All heavy logic lives in analysis/, services/, and ui/ packages.
"""

import logging
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config import INDEX_CONFIG, IST
from analysis import market_phase
from ui.components import inject_custom_css, phase_badge
from ui import tab_checklist, tab_oi_deep_dive, tab_stock_options, tab_market_breadth, tab_backtester, tab_news, tab_advisor

logging.basicConfig(level=logging.INFO)
# Set third-party loggers to WARNING to reduce noise
for _quiet in ("yfinance", "urllib3", "httpx", "httpcore", "curl_cffi"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Pre-Trade Checklist",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

inject_custom_css()

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    st.caption("No API keys required — all data from free public sources.")
    selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()), index=0)
    timeframe = st.selectbox("Intraday Interval", ["5m", "15m", "30m", "60m"], index=1)
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (market hours)", value=False)
    refresh_interval = st.select_slider(
        "Refresh interval",
        options=[30, 60, 90, 120, 180, 300],
        value=60,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x // 60}m" if x % 60 == 0 else f"{x // 60}m {x % 60}s",
        disabled=not auto_refresh,
    )
    refresh = st.button("Refresh Now", type="primary", width="stretch")

    st.divider()
    phase = market_phase()
    st.markdown(phase_badge(phase), unsafe_allow_html=True)
    now_ist = datetime.now(IST)
    st.caption(f"IST: {now_ist.strftime('%H:%M:%S')}")

    if auto_refresh and phase == "market_open":
        st.caption(f"Next refresh in ~{refresh_interval}s")

# Auto-refresh: JS-based timer — no toast spam, no rerun loop
if auto_refresh and phase == "market_open":
    st_autorefresh(interval=refresh_interval * 1000, limit=None, key="auto_refresh_timer")

display_name = INDEX_CONFIG[selected_index]["display_name"]

# ── Clear cache on manual refresh ──────────────────────────────────────────

if refresh:
    st.cache_data.clear()

# ══════════════════════════════════════════════════════════════════════════
# TABBED LAYOUT – each tab delegates to its own module
# ══════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    f"Pre-Trade Checklist ({display_name})",
    "OI Deep Dive",
    "Stock Options",
    "Market Breadth",
    "Market Advisor",
    "Strategy Backtester",
    "Market News",
])

with tabs[0]:
    tab_checklist.render(selected_index, display_name, timeframe)

with tabs[1]:
    tab_oi_deep_dive.render(selected_index, display_name)

with tabs[2]:
    tab_stock_options.render()

with tabs[3]:
    tab_market_breadth.render()

with tabs[4]:
    tab_advisor.render(selected_index, display_name, timeframe)

with tabs[5]:
    tab_backtester.render()

with tabs[6]:
    tab_news.render(selected_index, display_name)
