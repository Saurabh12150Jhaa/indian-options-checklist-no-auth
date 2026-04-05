"""
Indian Index Options Pre-Trade Checklist – Streamlit Dashboard
==============================================================
Zero-auth. Uses NSE India API + yfinance only.
Run with:  streamlit run app.py
"""

import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from config import INDEX_CONFIG, IST
from data_fetcher import (
    fetch_global_cues,
    fetch_india_vix,
    fetch_spot_data,
    fetch_historical,
    fetch_intraday,
    fetch_nse_option_chain,
    fetch_fii_dii,
)
from analysis import (
    run_technical_analysis,
    run_options_analysis,
    get_expiry_dates,
    generate_checklist,
    compute_overall_bias,
    market_phase,
    BULLISH,
    BEARISH,
    NEUTRAL,
)
from news_fetcher import fetch_news

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

st.markdown(
    """
    <style>
    .signal-bullish {
        background-color: #0e6b0e; color: white;
        padding: 4px 12px; border-radius: 6px; font-weight: 700;
        display: inline-block; text-align: center; min-width: 80px;
    }
    .signal-bearish {
        background-color: #b71c1c; color: white;
        padding: 4px 12px; border-radius: 6px; font-weight: 700;
        display: inline-block; text-align: center; min-width: 80px;
    }
    .signal-neutral {
        background-color: #f9a825; color: black;
        padding: 4px 12px; border-radius: 6px; font-weight: 700;
        display: inline-block; text-align: center; min-width: 80px;
    }
    .bias-box {
        padding: 24px; border-radius: 12px; text-align: center;
        font-size: 1.3rem; font-weight: 700; margin: 10px 0 20px;
    }
    .bias-bullish { background: linear-gradient(135deg, #1b5e20, #388e3c); color: white; }
    .bias-bearish { background: linear-gradient(135deg, #b71c1c, #e53935); color: white; }
    .bias-neutral { background: linear-gradient(135deg, #f57f17, #fbc02d); color: black; }
    .freshness { font-size: 0.75rem; color: #888; font-style: italic; }
    .phase-badge {
        padding: 4px 14px; border-radius: 20px; font-weight: 700;
        display: inline-block; font-size: 0.85rem;
    }
    .phase-pre_market  { background: #1565c0; color: white; }
    .phase-market_open { background: #2e7d32; color: white; }
    .phase-post_market { background: #616161; color: white; }
    div[data-testid="stMetric"] { background: #1e1e2f; border-radius: 10px; padding: 12px; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def signal_badge(sig: str) -> str:
    css = {"BULLISH": "signal-bullish", "BEARISH": "signal-bearish"}.get(sig, "signal-neutral")
    return f'<span class="{css}">{sig}</span>'


def bias_box_html(label: str, pct: float) -> str:
    css = {"BULLISH": "bias-bullish", "BEARISH": "bias-bearish"}.get(label, "bias-neutral")
    arrow = {"BULLISH": "&#9650;", "BEARISH": "&#9660;"}.get(label, "&#9654;")
    return (
        f'<div class="bias-box {css}">'
        f'{arrow} Overall Bias: {label} &nbsp; ({pct:+.1f}%)'
        f"</div>"
    )


def freshness(dt: datetime) -> str:
    return f'<span class="freshness">Updated: {dt.strftime("%d %b %Y, %H:%M:%S IST")}</span>'


def phase_badge(phase: str) -> str:
    labels = {"pre_market": "PRE-MARKET", "market_open": "MARKET OPEN", "post_market": "POST-MARKET"}
    return f'<span class="phase-badge phase-{phase}">{labels.get(phase, phase.upper())}</span>'


def colour_change(val):
    """Style helper for colouring positive/negative numbers."""
    if isinstance(val, (int, float)):
        c = "#4caf50" if val >= 0 else "#ef5350"
        return f"color: {c}; font-weight: 700"
    return ""


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

# ── Cached data loaders ───────────────────────────────────────────────────


@st.cache_data(ttl=180, show_spinner=False)
def _global_cues():
    return fetch_global_cues()


@st.cache_data(ttl=120, show_spinner=False)
def _india_vix():
    return fetch_india_vix()


@st.cache_data(ttl=60, show_spinner=False)
def _spot_data(idx):
    return fetch_spot_data(idx)


@st.cache_data(ttl=120, show_spinner=False)
def _historical(idx):
    return fetch_historical(idx)


@st.cache_data(ttl=60, show_spinner=False)
def _intraday(idx, interval):
    return fetch_intraday(idx, interval)


@st.cache_data(ttl=90, show_spinner=False)
def _nse_option_chain(idx):
    return fetch_nse_option_chain(idx)


@st.cache_data(ttl=300, show_spinner=False)
def _fii_dii():
    return fetch_fii_dii()


@st.cache_data(ttl=120, show_spinner=False)
def _news(idx):
    return fetch_news(idx)


if refresh:
    st.cache_data.clear()

# ══════════════════════════════════════════════════════════════════════════
# TABBED LAYOUT
# ══════════════════════════════════════════════════════════════════════════

tab_checklist, tab_oi_deep, tab_stocks, tab_breadth, tab_backtest, tab_news = st.tabs([
    f"Pre-Trade Checklist ({display_name})",
    "OI Deep Dive",
    "Stock Options",
    "Market Breadth",
    "Strategy Backtester",
    "Market News",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 – PRE-TRADE CHECKLIST (existing content)
# ══════════════════════════════════════════════════════════════════════════

with tab_checklist:
    # SECTION 1 – GLOBAL CUES & MACRO
    st.header("1. Global Cues & Macro")
    col_global, col_vix, col_fii = st.columns([2, 1, 1.5])

    with col_global:
        with st.spinner("Fetching global indices..."):
            global_df, global_ts = _global_cues()
        if global_df is not None and not global_df.empty:
            styled = global_df.style.map(colour_change, subset=["Change %"])
            st.dataframe(styled, width="stretch", hide_index=True)
            st.markdown(freshness(global_ts), unsafe_allow_html=True)
        else:
            st.warning("Global cues unavailable.")

    with col_vix:
        st.subheader("India VIX")
        with st.spinner("Fetching VIX..."):
            vix_value, vix_ts = _india_vix()
        if vix_value is not None:
            delta_color = "inverse" if vix_value > 18 else "normal"
            st.metric("VIX", f"{vix_value:.2f}", delta_color=delta_color)
            st.markdown(freshness(vix_ts), unsafe_allow_html=True)
        else:
            st.metric("VIX", "N/A")

    with col_fii:
        st.subheader("FII / DII Activity")
        with st.spinner("Fetching FII/DII..."):
            fii_dii_df, fii_ts = _fii_dii()
        if fii_dii_df is not None and not fii_dii_df.empty:
            st.dataframe(fii_dii_df, width="stretch", hide_index=True)
            st.markdown(freshness(fii_ts), unsafe_allow_html=True)
        else:
            st.caption("FII/DII data unavailable (NSE may restrict automated access).")

    # SECTION 2 – SPOT & TECHNICAL ANALYSIS
    st.header(f"2. {display_name} – Spot & Technical Analysis")

    with st.spinner("Fetching spot & historical data..."):
        spot_info, spot_ts = _spot_data(selected_index)
        hist_df, hist_ts = _historical(selected_index)
        intra_df, intra_ts = _intraday(selected_index, timeframe)

    # Spot metrics
    if spot_info:
        ltp = spot_info["ltp"]
        prev_close = spot_info["prev_close"]
        change = ltp - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spot Price", f"{ltp:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
        m2.metric("Open", f"{spot_info['open']:,.2f}")
        m3.metric("High", f"{spot_info['high']:,.2f}")
        m4.metric("Low", f"{spot_info['low']:,.2f}")
        m5.metric("Prev Close", f"{prev_close:,.2f}")
        st.markdown(freshness(spot_ts), unsafe_allow_html=True)
    else:
        st.warning("Could not fetch spot data from yfinance.")
        ltp = 0

    # Technicals
    technicals: dict = {}
    if hist_df is not None and not hist_df.empty:
        technicals = run_technical_analysis(hist_df)

        # ── Interactive Technical Chart with all overlays ──────────────
        st.subheader("Technical Analysis Chart")

        chart_source = st.radio(
            "Chart data",
            ["Intraday", "Daily"],
            horizontal=True,
            index=0 if (intra_df is not None and not intra_df.empty) else 1,
            key="chart_source",
        )

        if chart_source == "Intraday" and intra_df is not None and not intra_df.empty:
            chart_df = intra_df.copy()
            chart_title = f"{display_name} — Intraday ({timeframe})"
        else:
            chart_df = hist_df.tail(120).copy()
            chart_title = f"{display_name} — Daily (last 120 sessions)"

        # Build the main chart with subplots: candlestick + RSI + MACD
        tech_fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("", "RSI (14)", "MACD"),
        )

        # 1) Candlestick
        tech_fig.add_trace(
            go.Candlestick(
                x=chart_df.index, open=chart_df["open"], high=chart_df["high"],
                low=chart_df["low"], close=chart_df["close"],
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                name="Price", showlegend=False,
            ),
            row=1, col=1,
        )

        # 2) EMA overlays
        import pandas_ta as _ta
        ema_colours = {9: "#fdd835", 20: "#42a5f5", 50: "#ab47bc", 200: "#ef6c00"}
        for period in [9, 20, 50, 200]:
            ema_series = _ta.ema(chart_df["close"], length=period)
            if ema_series is not None and not ema_series.dropna().empty:
                tech_fig.add_trace(
                    go.Scatter(
                        x=chart_df.index, y=ema_series,
                        mode="lines", name=f"EMA {period}",
                        line=dict(color=ema_colours.get(period, "#ffffff"), width=1.2),
                    ),
                    row=1, col=1,
                )

        # 3) Pivot levels (horizontal lines on price chart)
        pivots = technicals.get("pivots")
        if pivots:
            pivot_colors = {
                "PP": "#ffffff", "R1": "#ef9a9a", "R2": "#ef5350", "R3": "#b71c1c",
                "S1": "#a5d6a7", "S2": "#66bb6a", "S3": "#1b5e20",
            }
            for level, price in pivots.items():
                tech_fig.add_hline(
                    y=price, line_dash="dot",
                    line_color=pivot_colors.get(level, "#888"),
                    opacity=0.5, row=1, col=1,
                    annotation_text=f"{level}: {price:,.0f}",
                    annotation_position="right",
                    annotation_font_size=9,
                    annotation_font_color=pivot_colors.get(level, "#888"),
                )

        # 4) Fibonacci levels (horizontal bands)
        fib = technicals.get("fibonacci")
        if fib:
            fib_colors = {
                "0.236": "#4fc3f7", "0.382": "#29b6f6", "0.500": "#0288d1",
                "0.618": "#01579b", "0.786": "#002171",
            }
            for level, price in fib.items():
                if level in fib_colors:
                    tech_fig.add_hline(
                        y=price, line_dash="dash",
                        line_color=fib_colors[level], opacity=0.4, row=1, col=1,
                        annotation_text=f"Fib {level}: {price:,.0f}",
                        annotation_position="left",
                        annotation_font_size=8,
                        annotation_font_color=fib_colors[level],
                    )

        # 5) RSI subplot
        rsi_series = _ta.rsi(chart_df["close"], length=14)
        if rsi_series is not None and not rsi_series.dropna().empty:
            tech_fig.add_trace(
                go.Scatter(x=chart_df.index, y=rsi_series, mode="lines", name="RSI",
                           line=dict(color="#ab47bc", width=1.5)),
                row=2, col=1,
            )
            tech_fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.5, row=2, col=1)
            tech_fig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", opacity=0.5, row=2, col=1)
            tech_fig.add_hrect(y0=30, y1=70, fillcolor="#fdd835", opacity=0.05, row=2, col=1)

        # 6) MACD subplot
        macd_result = _ta.macd(chart_df["close"], fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.dropna().empty:
            macd_col = [c for c in macd_result.columns if c.startswith("MACD_") and not c.startswith("MACDs") and not c.startswith("MACDh")]
            signal_col = [c for c in macd_result.columns if c.startswith("MACDs")]
            hist_col = [c for c in macd_result.columns if c.startswith("MACDh")]

            if macd_col:
                tech_fig.add_trace(
                    go.Scatter(x=chart_df.index, y=macd_result[macd_col[0]], mode="lines",
                               name="MACD", line=dict(color="#42a5f5", width=1.2)),
                    row=3, col=1,
                )
            if signal_col:
                tech_fig.add_trace(
                    go.Scatter(x=chart_df.index, y=macd_result[signal_col[0]], mode="lines",
                               name="Signal", line=dict(color="#ef6c00", width=1.2)),
                    row=3, col=1,
                )
            if hist_col:
                hist_vals = macd_result[hist_col[0]]
                colours = ["#66bb6a" if v >= 0 else "#ef5350" for v in hist_vals]
                tech_fig.add_trace(
                    go.Bar(x=chart_df.index, y=hist_vals, name="Histogram",
                           marker_color=colours, opacity=0.6),
                    row=3, col=1,
                )

        # Layout
        tech_fig.update_layout(
            title=chart_title,
            height=700,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font_size=10),
            hovermode="x unified",
        )
        tech_fig.update_yaxes(title_text="Price", row=1, col=1)
        tech_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        tech_fig.update_yaxes(title_text="MACD", row=3, col=1)

        st.plotly_chart(tech_fig, width="stretch", key="tech_chart")

        # ── Summary cards below chart ─────────────────────────────────
        tc1, tc2, tc3, tc4 = st.columns(4)

        with tc1:
            st.markdown("**Moving Averages**")
            ema_rows = []
            for period, val in technicals["emas"].items():
                pos = "Above" if technicals["spot"] > val else "Below"
                ema_rows.append({"EMA": f"EMA {period}", "Value": f"{val:,.2f}", "Spot": pos})
            if ema_rows:
                ema_df = pd.DataFrame(ema_rows)
                st.dataframe(ema_df, width="stretch", hide_index=True, height=180)

        with tc2:
            st.markdown("**Momentum**")
            rsi_val = technicals["rsi"]
            rsi_colour = "#66bb6a" if rsi_val and rsi_val < 40 else "#ef5350" if rsi_val and rsi_val > 60 else "#fdd835"
            st.markdown(
                f'<div style="text-align:center; padding:8px; background:#1e1e2f; border-radius:8px; margin-bottom:6px;">'
                f'<span style="color:#aaa; font-size:0.85rem;">RSI (14)</span><br/>'
                f'<span style="color:{rsi_colour}; font-size:1.5rem; font-weight:700;">{rsi_val if rsi_val else "N/A"}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            macd_d = technicals["macd"]
            if macd_d:
                macd_colour = "#66bb6a" if macd_d["histogram"] > 0 else "#ef5350"
                st.markdown(
                    f'<div style="text-align:center; padding:8px; background:#1e1e2f; border-radius:8px;">'
                    f'<span style="color:#aaa; font-size:0.85rem;">MACD Histogram</span><br/>'
                    f'<span style="color:{macd_colour}; font-size:1.5rem; font-weight:700;">{macd_d["histogram"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with tc3:
            st.markdown("**Pivot Points (Classic)**")
            if pivots:
                p_rows = [{"Level": k, "Price": f"{v:,.2f}"} for k, v in pivots.items()]
                st.dataframe(pd.DataFrame(p_rows), width="stretch", hide_index=True, height=180)

        with tc4:
            st.markdown("**Fibonacci Retracement**")
            if fib:
                f_rows = [{"Level": k, "Price": f"{v:,.2f}"} for k, v in fib.items()]
                st.dataframe(pd.DataFrame(f_rows), width="stretch", hide_index=True, height=180)

        st.markdown(freshness(hist_ts), unsafe_allow_html=True)
    else:
        st.warning("Historical data unavailable – technical analysis skipped.")

    # SECTION 3 – OPTIONS CHAIN ANALYSIS (NSE)
    st.header(f"3. {display_name} – Options Chain Analysis")

    with st.spinner("Fetching NSE option chain (may take a moment)..."):
        nse_data, oc_ts = _nse_option_chain(selected_index)

    if not nse_data:
        st.warning(
            "Could not fetch option chain from NSE. "
            "NSE may be blocking automated requests — try refreshing in a few seconds."
        )
        options_result = {
            "spot": ltp, "chain_df": pd.DataFrame(),
            "pcr_oi": None, "pcr_volume": None, "max_pain": None,
            "highest_call_oi_strike": None, "highest_put_oi_strike": None,
            "highest_call_oi_value": None, "highest_put_oi_value": None,
            "call_buildup": 0, "put_buildup": 0, "buildup_signal": NEUTRAL,
            "atm_call_iv": 0, "atm_put_iv": 0, "iv_skew": 0,
            "atm_straddle": None,
        }
        selected_expiry = None
    else:
        expiry_list = get_expiry_dates(nse_data)
        selected_expiry = st.selectbox("Expiry Date", expiry_list, index=0, key="checklist_expiry") if expiry_list else None

        options_result = run_options_analysis(nse_data, selected_expiry)
        spot_for_oc = options_result["spot"] or ltp
        chain_df: pd.DataFrame = options_result["chain_df"]

        # Save OI snapshot for tracking
        try:
            from oi_tracker import save_oi_snapshot
            if not chain_df.empty and selected_expiry:
                save_oi_snapshot(selected_index, selected_expiry, chain_df, spot_for_oc)
        except Exception:
            pass

        # Key options metrics
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("PCR (OI)", options_result["pcr_oi"] if options_result["pcr_oi"] else "N/A")
        o2.metric("PCR (Volume)", options_result["pcr_volume"] if options_result["pcr_volume"] else "N/A")
        o3.metric("Max Pain", f"{options_result['max_pain']:,.0f}" if options_result["max_pain"] else "N/A")
        o4.metric("Spot (NSE)", f"{spot_for_oc:,.2f}")

        oi1, oi2, oi3, oi4 = st.columns(4)
        oi1.metric(
            "Highest Call OI (Resistance)",
            f"{options_result['highest_call_oi_strike']:,.0f}" if options_result.get("highest_call_oi_strike") else "N/A",
            f"OI: {options_result.get('highest_call_oi_value', 0):,}" if options_result.get("highest_call_oi_value") else None,
        )
        oi2.metric(
            "Highest Put OI (Support)",
            f"{options_result['highest_put_oi_strike']:,.0f}" if options_result.get("highest_put_oi_strike") else "N/A",
            f"OI: {options_result.get('highest_put_oi_value', 0):,}" if options_result.get("highest_put_oi_value") else None,
        )
        oi3.metric("ATM Call IV", f"{options_result['atm_call_iv']:.2f}%")
        oi4.metric("ATM Put IV", f"{options_result['atm_put_iv']:.2f}%")

        # ATM Straddle / Expected Move
        straddle = options_result.get("atm_straddle")
        if straddle:
            st.subheader("ATM Straddle – Expected Move")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("ATM Strike", f"{straddle['atm_strike']:,.0f}")
            s2.metric("Straddle Price", f"{straddle['straddle_price']:,.2f}")
            s3.metric("Expected Move", f"{straddle['expected_move_pct']:.2f}%")
            s4.metric("Range", f"{straddle['lower_range']:,.0f} – {straddle['upper_range']:,.0f}")

        st.markdown(freshness(oc_ts), unsafe_allow_html=True)

        # OI Distribution charts
        if not chain_df.empty:
            st.subheader("Open Interest Distribution")
            strike_gap = INDEX_CONFIG[selected_index]["strike_gap"]
            band = strike_gap * 20
            near = chain_df[
                (chain_df["strike"] >= spot_for_oc - band) & (chain_df["strike"] <= spot_for_oc + band)
            ]
            if near.empty:
                near = chain_df

            oi_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Open Interest by Strike", "OI Change by Strike"),
                horizontal_spacing=0.08,
            )
            oi_fig.add_trace(
                go.Bar(x=near["strike"], y=near["call_oi"], name="Call OI", marker_color="#ef5350", opacity=0.85),
                row=1, col=1,
            )
            oi_fig.add_trace(
                go.Bar(x=near["strike"], y=near["put_oi"], name="Put OI", marker_color="#26a69a", opacity=0.85),
                row=1, col=1,
            )
            oi_fig.add_trace(
                go.Bar(x=near["strike"], y=near["call_chg_oi"], name="Call OI Chg", marker_color="#ef5350", opacity=0.7),
                row=1, col=2,
            )
            oi_fig.add_trace(
                go.Bar(x=near["strike"], y=near["put_chg_oi"], name="Put OI Chg", marker_color="#26a69a", opacity=0.7),
                row=1, col=2,
            )
            for ci in [1, 2]:
                oi_fig.add_vline(x=spot_for_oc, line_dash="dash", line_color="yellow", opacity=0.6, row=1, col=ci)

            oi_fig.update_layout(
                barmode="group", height=420,
                margin=dict(l=40, r=20, t=40, b=30), template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(oi_fig, width="stretch")

            # IV smile
            st.subheader("Implied Volatility Smile")
            iv_fig = go.Figure()
            iv_fig.add_trace(go.Scatter(x=near["strike"], y=near["call_iv"], mode="lines+markers", name="Call IV", line=dict(color="#ef5350")))
            iv_fig.add_trace(go.Scatter(x=near["strike"], y=near["put_iv"], mode="lines+markers", name="Put IV", line=dict(color="#26a69a")))
            iv_fig.add_vline(x=spot_for_oc, line_dash="dash", line_color="yellow", opacity=0.6)
            iv_fig.update_layout(
                xaxis_title="Strike", yaxis_title="Implied Volatility (%)",
                height=350, margin=dict(l=40, r=20, t=30, b=30), template="plotly_dark",
            )
            st.plotly_chart(iv_fig, width="stretch")

    # SECTION 4 – PRE-TRADE CHECKLIST & OVERALL BIAS
    st.header("4. Pre-Trade Checklist Summary")

    checklist = generate_checklist(
        technicals=technicals if technicals else {"spot": ltp, "emas": {}, "rsi": None, "macd": None},
        options_data=options_result,
        vix=vix_value,
        global_df=global_df if global_df is not None else pd.DataFrame(),
        fii_dii_df=fii_dii_df,
    )

    overall_label, overall_pct = compute_overall_bias(checklist)

    # ── Bias box + gauge side by side ─────────────────────────────────
    bias_c1, bias_c2 = st.columns([1.2, 1])
    with bias_c1:
        st.markdown(bias_box_html(overall_label, overall_pct), unsafe_allow_html=True)
    with bias_c2:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=overall_pct,
                title={"text": "Bias Score", "font": {"size": 14}},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [-100, 100], "tickwidth": 1},
                    "bar": {"color": "#1e88e5"},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [-100, -25], "color": "#ef5350"},
                        {"range": [-25, 25], "color": "#fdd835"},
                        {"range": [25, 100], "color": "#66bb6a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.8,
                        "value": overall_pct,
                    },
                },
            )
        )
        gauge_fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), template="plotly_dark")
        st.plotly_chart(gauge_fig, width="stretch")

    # ── Checklist as styled HTML table ────────────────────────────────
    def _build_checklist_html(items: list[dict]) -> str:
        """Build a clean HTML table for the checklist."""
        signal_styles = {
            "BULLISH": "background:#1b5e20; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;",
            "BEARISH": "background:#b71c1c; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;",
            "NEUTRAL": "background:#f57f17; color:#000; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;",
        }
        weight_bar_max = max(i["weight"] for i in items) if items else 1

        rows_html = ""
        for i, item in enumerate(items):
            sig = item["signal"]
            sig_html = f'<span style="{signal_styles.get(sig, signal_styles["NEUTRAL"])}">{sig}</span>'

            # Weight as a visual bar
            bar_pct = (item["weight"] / weight_bar_max) * 100
            bar_colour = "#42a5f5"
            weight_html = (
                f'<div style="display:flex; align-items:center; gap:6px;">'
                f'<div style="background:{bar_colour}; height:8px; width:{bar_pct:.0f}%; border-radius:4px; min-width:10px;"></div>'
                f'<span style="color:#ccc; font-size:0.8rem;">{item["weight"]:.1f}</span>'
                f'</div>'
            )

            # Row background alternation
            bg = "#1a1a2e" if i % 2 == 0 else "#16213e"

            # Value styling — colour hint based on signal
            val_color = {"BULLISH": "#66bb6a", "BEARISH": "#ef5350"}.get(sig, "#e0e0e0")

            rows_html += (
                f'<tr style="background:{bg};">'
                f'<td style="padding:10px 14px; font-weight:600; color:#e0e0e0; border-bottom:1px solid #2a2a4a;">'
                f'<span style="color:#aaa; font-size:0.75rem; margin-right:6px;">#{i+1}</span>{item["indicator"]}</td>'
                f'<td style="padding:10px 14px; color:{val_color}; font-weight:500; border-bottom:1px solid #2a2a4a;">{item["value"]}</td>'
                f'<td style="padding:10px 14px; border-bottom:1px solid #2a2a4a;">{sig_html}</td>'
                f'<td style="padding:10px 14px; border-bottom:1px solid #2a2a4a;">{weight_html}</td>'
                f'</tr>'
            )

        return (
            f'<table style="width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden; '
            f'border:1px solid #2a2a4a; margin-top:8px;">'
            f'<thead>'
            f'<tr style="background:#0d1b2a;">'
            f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Indicator</th>'
            f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Value</th>'
            f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Signal</th>'
            f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Weight</th>'
            f'</tr>'
            f'</thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>'
        )

    st.markdown(_build_checklist_html(checklist), unsafe_allow_html=True)

    # ── Signal summary bar ────────────────────────────────────────────
    bull_count = sum(1 for c in checklist if c["signal"] == BULLISH)
    bear_count = sum(1 for c in checklist if c["signal"] == BEARISH)
    neut_count = sum(1 for c in checklist if c["signal"] == NEUTRAL)
    total_checks = len(checklist)

    bull_pct = (bull_count / total_checks * 100) if total_checks else 0
    bear_pct = (bear_count / total_checks * 100) if total_checks else 0
    neut_pct = (neut_count / total_checks * 100) if total_checks else 0

    st.markdown(
        f'<div style="display:flex; height:24px; border-radius:6px; overflow:hidden; margin:12px 0 4px;">'
        f'<div style="width:{bull_pct}%; background:#388e3c;" title="Bullish: {bull_count}"></div>'
        f'<div style="width:{neut_pct}%; background:#f9a825;" title="Neutral: {neut_count}"></div>'
        f'<div style="width:{bear_pct}%; background:#c62828;" title="Bearish: {bear_count}"></div>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#aaa;">'
        f'<span style="color:#66bb6a;">Bullish: {bull_count}/{total_checks}</span>'
        f'<span style="color:#fdd835;">Neutral: {neut_count}/{total_checks}</span>'
        f'<span style="color:#ef5350;">Bearish: {bear_count}/{total_checks}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 – OI DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════

with tab_oi_deep:
    st.header(f"OI Deep Dive – {display_name}")

    try:
        from oi_tracker import get_oi_timeline, get_aggregate_oi_timeline, get_tracked_dates
        from analysis import parse_nse_option_chain, get_expiry_dates as get_expiries

        if nse_data:
            expiry_list_deep = get_expiries(nse_data)
            d1, d2 = st.columns(2)
            with d1:
                deep_expiry = st.selectbox("Expiry", expiry_list_deep, index=0, key="deep_expiry")
            with d2:
                tracked = get_tracked_dates(selected_index)
                track_info = f"Tracking {len(tracked)} date(s)" if tracked else "No historical OI data yet"
                st.caption(track_info)

            # Multi-expiry OI comparison
            st.subheader("Multi-Expiry OI Comparison")
            chain_dfs = {}
            for exp in expiry_list_deep[:4]:
                cdf = parse_nse_option_chain(nse_data, exp)
                if not cdf.empty:
                    chain_dfs[exp] = cdf

            if chain_dfs:
                exp_fig = make_subplots(rows=1, cols=2, subplot_titles=("Call OI by Expiry", "Put OI by Expiry"))
                colors = ["#ef5350", "#ff7043", "#ffb74d", "#fff176"]
                for i, (exp, cdf) in enumerate(chain_dfs.items()):
                    spot_val = options_result.get("spot", ltp) or ltp
                    sg = INDEX_CONFIG[selected_index]["strike_gap"]
                    near_exp = cdf[(cdf["strike"] >= spot_val - sg * 15) & (cdf["strike"] <= spot_val + sg * 15)]
                    if near_exp.empty:
                        near_exp = cdf
                    exp_fig.add_trace(
                        go.Bar(x=near_exp["strike"], y=near_exp["call_oi"], name=f"CE {exp}", marker_color=colors[i % 4], opacity=0.7),
                        row=1, col=1,
                    )
                    green_shades = ["#26a69a", "#66bb6a", "#a5d6a7", "#c8e6c9"]
                    exp_fig.add_trace(
                        go.Bar(x=near_exp["strike"], y=near_exp["put_oi"], name=f"PE {exp}", marker_color=green_shades[i % 4], opacity=0.7),
                        row=1, col=2,
                    )
                exp_fig.update_layout(barmode="group", height=420, template="plotly_dark",
                                      margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(exp_fig, width="stretch")

            # OI Timeline (from tracker)
            st.subheader("OI Change Timeline (Intraday)")
            agg_tl = get_aggregate_oi_timeline(selected_index, deep_expiry)
            if not agg_tl.empty:
                tl_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=("Total OI", "PCR Over Time"), vertical_spacing=0.12)
                tl_fig.add_trace(go.Scatter(x=agg_tl["timestamp"], y=agg_tl["total_call_oi"], name="Call OI", line=dict(color="#ef5350")), row=1, col=1)
                tl_fig.add_trace(go.Scatter(x=agg_tl["timestamp"], y=agg_tl["total_put_oi"], name="Put OI", line=dict(color="#26a69a")), row=1, col=1)
                tl_fig.add_trace(go.Scatter(x=agg_tl["timestamp"], y=agg_tl["pcr"], name="PCR", line=dict(color="#42a5f5")), row=2, col=1)
                tl_fig.add_hline(y=1.0, line_dash="dash", line_color="yellow", opacity=0.5, row=2, col=1)
                tl_fig.update_layout(height=500, template="plotly_dark", margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(tl_fig, width="stretch")
            else:
                st.info("OI timeline data will appear after multiple refreshes capture snapshots over time.")

            # ITM/OTM Analysis
            if nse_data and deep_expiry:
                deep_chain = parse_nse_option_chain(nse_data, deep_expiry)
                spot_val = options_result.get("spot", ltp) or ltp
                if not deep_chain.empty and spot_val > 0:
                    st.subheader("ITM / OTM Analysis")
                    itm_calls = deep_chain[deep_chain["strike"] < spot_val]
                    otm_calls = deep_chain[deep_chain["strike"] >= spot_val]
                    itm_puts = deep_chain[deep_chain["strike"] > spot_val]
                    otm_puts = deep_chain[deep_chain["strike"] <= spot_val]

                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("ITM Call OI", f"{itm_calls['call_oi'].sum():,.0f}")
                    ic2.metric("OTM Call OI", f"{otm_calls['call_oi'].sum():,.0f}")
                    ic3.metric("ITM Put OI", f"{itm_puts['put_oi'].sum():,.0f}")
                    ic4.metric("OTM Put OI", f"{otm_puts['put_oi'].sum():,.0f}")
        else:
            st.info("Option chain data needed. Go to Pre-Trade Checklist tab to load data first.")
    except ImportError:
        st.info("OI tracker module not available. It will be created automatically.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 – STOCK OPTIONS
# ══════════════════════════════════════════════════════════════════════════

with tab_stocks:
    st.header("Stock Options Analysis")

    try:
        from config import FNO_STOCKS
        from market_data import fetch_stock_option_chain
        from analysis import run_options_analysis as run_opts, get_expiry_dates as get_exp

        stock_symbol = st.selectbox("Select F&O Stock", list(FNO_STOCKS.keys()), key="stock_picker")
        stock_cfg = FNO_STOCKS[stock_symbol]

        with st.spinner(f"Fetching {stock_symbol} option chain..."):
            stock_oc_data, stock_oc_ts = fetch_stock_option_chain(stock_symbol)

        if stock_oc_data:
            stock_expiries = get_exp(stock_oc_data)
            stock_expiry = st.selectbox("Expiry", stock_expiries, index=0, key="stock_expiry") if stock_expiries else None

            stock_opts = run_opts(stock_oc_data, stock_expiry)
            stock_spot = stock_opts["spot"]
            stock_chain = stock_opts["chain_df"]

            # Key metrics
            so1, so2, so3, so4 = st.columns(4)
            so1.metric("Spot", f"{stock_spot:,.2f}" if stock_spot else "N/A")
            so2.metric("PCR (OI)", stock_opts["pcr_oi"] if stock_opts["pcr_oi"] else "N/A")
            so3.metric("Max Pain", f"{stock_opts['max_pain']:,.0f}" if stock_opts["max_pain"] else "N/A")
            so4.metric("Lot Size", stock_cfg["lot_size"])

            so5, so6, so7, so8 = st.columns(4)
            so5.metric("Resistance (Call OI)",
                       f"{stock_opts['highest_call_oi_strike']:,.0f}" if stock_opts.get("highest_call_oi_strike") else "N/A")
            so6.metric("Support (Put OI)",
                       f"{stock_opts['highest_put_oi_strike']:,.0f}" if stock_opts.get("highest_put_oi_strike") else "N/A")
            so7.metric("ATM Call IV", f"{stock_opts['atm_call_iv']:.2f}%")
            so8.metric("ATM Put IV", f"{stock_opts['atm_put_iv']:.2f}%")

            straddle = stock_opts.get("atm_straddle")
            if straddle:
                st.subheader("ATM Straddle")
                ss1, ss2, ss3 = st.columns(3)
                ss1.metric("Straddle Price", f"{straddle['straddle_price']:,.2f}")
                ss2.metric("Expected Move", f"{straddle['expected_move_pct']:.2f}%")
                ss3.metric("Range", f"{straddle['lower_range']:,.0f} – {straddle['upper_range']:,.0f}")

            st.markdown(freshness(stock_oc_ts), unsafe_allow_html=True)

            # OI chart
            if not stock_chain.empty:
                st.subheader("OI Distribution")
                sg = stock_cfg["strike_gap"]
                band = sg * 15
                near_stock = stock_chain[
                    (stock_chain["strike"] >= stock_spot - band) & (stock_chain["strike"] <= stock_spot + band)
                ]
                if near_stock.empty:
                    near_stock = stock_chain

                soi_fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=("Open Interest", "OI Change"),
                                        horizontal_spacing=0.08)
                soi_fig.add_trace(go.Bar(x=near_stock["strike"], y=near_stock["call_oi"], name="Call OI", marker_color="#ef5350"), row=1, col=1)
                soi_fig.add_trace(go.Bar(x=near_stock["strike"], y=near_stock["put_oi"], name="Put OI", marker_color="#26a69a"), row=1, col=1)
                soi_fig.add_trace(go.Bar(x=near_stock["strike"], y=near_stock["call_chg_oi"], name="Call Chg", marker_color="#ef5350", opacity=0.7), row=1, col=2)
                soi_fig.add_trace(go.Bar(x=near_stock["strike"], y=near_stock["put_chg_oi"], name="Put Chg", marker_color="#26a69a", opacity=0.7), row=1, col=2)
                for ci in [1, 2]:
                    soi_fig.add_vline(x=stock_spot, line_dash="dash", line_color="yellow", opacity=0.6, row=1, col=ci)
                soi_fig.update_layout(barmode="group", height=420, template="plotly_dark",
                                      margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(soi_fig, width="stretch")
        else:
            st.warning(f"Could not fetch option chain for {stock_symbol}. NSE may be rate-limiting.")
    except ImportError as e:
        st.info(f"Stock options module loading: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 – MARKET BREADTH
# ══════════════════════════════════════════════════════════════════════════

with tab_breadth:
    st.header("Market Breadth")

    try:
        from market_data import (
            fetch_top_gainers, fetch_top_losers,
            fetch_advances_declines, fetch_pre_open,
        )

        # Advances / Declines
        st.subheader("Advances vs Declines")
        with st.spinner("Fetching market breadth..."):
            ad_data, ad_ts = fetch_advances_declines()
        if ad_data:
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Advances", ad_data["advances"])
            bc2.metric("Declines", ad_data["declines"])
            bc3.metric("Unchanged", ad_data["unchanged"])
            bc4.metric("A/D Ratio", ad_data["ad_ratio"])

            ad_fig = go.Figure(data=[
                go.Pie(
                    labels=["Advances", "Declines", "Unchanged"],
                    values=[ad_data["advances"], ad_data["declines"], ad_data["unchanged"]],
                    marker=dict(colors=["#66bb6a", "#ef5350", "#fdd835"]),
                    hole=0.4,
                )
            ])
            ad_fig.update_layout(height=300, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(ad_fig, width="stretch")
            st.markdown(freshness(ad_ts), unsafe_allow_html=True)
        else:
            st.info("Market breadth data unavailable.")

        # Top Gainers / Losers
        g_col, l_col = st.columns(2)

        with g_col:
            st.subheader("Top Gainers")
            with st.spinner("Loading..."):
                gainers_df, g_ts = fetch_top_gainers()
            if gainers_df is not None and not gainers_df.empty:
                st.dataframe(gainers_df, width="stretch", hide_index=True)
            else:
                st.info("Top gainers unavailable.")

        with l_col:
            st.subheader("Top Losers")
            with st.spinner("Loading..."):
                losers_df, l_ts = fetch_top_losers()
            if losers_df is not None and not losers_df.empty:
                st.dataframe(losers_df, width="stretch", hide_index=True)
            else:
                st.info("Top losers unavailable.")

        # Pre-open data
        st.subheader("Pre-Open Market")
        po_key = st.selectbox("Pre-Open Index", ["NIFTY", "BANKNIFTY", "FO"], key="preopen_key")
        with st.spinner("Loading pre-open data..."):
            po_df, po_ts = fetch_pre_open(po_key)
        if po_df is not None and not po_df.empty:
            st.dataframe(po_df.head(20), width="stretch", hide_index=True)
            st.markdown(freshness(po_ts), unsafe_allow_html=True)
        else:
            st.info("Pre-open data unavailable (available 9:00-9:15 IST).")

    except ImportError as e:
        st.info(f"Market data module loading: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 – STRATEGY BACKTESTER
# ══════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.header("Strategy Backtester")

    try:
        from backtester.strategies import PREBUILT_STRATEGIES, get_strategy_fn
        from backtester.smart_money import SMC_STRATEGIES
        from backtester.price_action import PA_STRATEGIES
        from backtester.custom_strategy import CustomStrategy, CUSTOM_PRESETS, CONDITION_REGISTRY, LEG_TEMPLATES

        # Combine all strategies
        all_strategies = {}
        for name, info in PREBUILT_STRATEGIES.items():
            all_strategies[name] = {**info, "source": "prebuilt"}
        for name, info in SMC_STRATEGIES.items():
            all_strategies[name] = {**info, "source": "smc"}
        for name, info in PA_STRATEGIES.items():
            all_strategies[name] = {**info, "source": "price_action"}
        for name, preset in CUSTOM_PRESETS.items():
            all_strategies[name] = {
                "fn": CustomStrategy(preset),
                "category": "Custom Preset",
                "description": f"Custom: {', '.join(c['type'] for c in preset.get('conditions', []))}",
                "source": "custom",
            }

        # Strategy selection
        st.subheader("Select Strategy")
        categories = sorted(set(s["category"] for s in all_strategies.values()))
        sel_category = st.selectbox("Category", ["All"] + categories, key="bt_cat")

        filtered_strats = {k: v for k, v in all_strategies.items()
                          if sel_category == "All" or v["category"] == sel_category}

        sel_strategy = st.selectbox(
            "Strategy",
            list(filtered_strats.keys()),
            key="bt_strat",
            format_func=lambda x: f"{x} ({filtered_strats[x]['category']})",
        )

        if sel_strategy:
            st.info(f"**{sel_strategy}**: {filtered_strats[sel_strategy]['description']}")

        st.divider()

        # Custom strategy builder
        with st.expander("Build Custom Strategy", expanded=False):
            st.subheader("Custom Strategy Builder")
            custom_name = st.text_input("Strategy Name", "My Custom Strategy", key="custom_name")

            # Entry conditions
            st.markdown("**Entry Conditions:**")
            cond_logic = st.radio("Condition Logic", ["AND", "OR"], horizontal=True, key="cond_logic")
            num_conditions = st.number_input("Number of conditions", 1, 5, 2, key="num_conds")

            conditions = []
            for i in range(int(num_conditions)):
                st.markdown(f"**Condition {i+1}:**")
                cc1, cc2 = st.columns(2)
                with cc1:
                    cond_type = st.selectbox(
                        f"Type",
                        list(CONDITION_REGISTRY.keys()),
                        key=f"cond_type_{i}",
                        format_func=lambda x: CONDITION_REGISTRY[x]["label"],
                    )
                cond_info = CONDITION_REGISTRY[cond_type]
                params = {}
                with cc2:
                    for p in cond_info["params"]:
                        pk = f"cond_{i}_{p['name']}"
                        if p["type"] == "int":
                            params[p["name"]] = st.number_input(p["label"], p.get("min", 1), p.get("max", 200), p["default"], key=pk)
                        elif p["type"] == "float":
                            params[p["name"]] = st.number_input(p["label"], float(p.get("min", 0.0)), float(p.get("max", 100.0)), float(p["default"]), step=0.1, key=pk)
                        elif p["type"] == "select":
                            params[p["name"]] = st.selectbox(p["label"], p["options"], index=p["options"].index(p["default"]), key=pk)
                        elif p["type"] == "multiselect":
                            params[p["name"]] = st.multiselect(p["label"], p["options"], default=p["default"], key=pk)
                conditions.append({"type": cond_type, "params": params})

            # Leg builder
            st.markdown("**Option Legs:**")
            num_legs = st.number_input("Number of legs", 1, 4, 2, key="num_legs")
            legs = []
            for i in range(int(num_legs)):
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    template = st.selectbox(
                        f"Leg {i+1} Strike",
                        list(LEG_TEMPLATES.keys()),
                        key=f"leg_tmpl_{i}",
                        format_func=lambda x: LEG_TEMPLATES[x]["label"],
                    )
                with lc2:
                    action = st.selectbox(f"Action", ["BUY", "SELL"], key=f"leg_action_{i}")
                with lc3:
                    qty = st.number_input(f"Qty", 1, 10, 1, key=f"leg_qty_{i}")
                legs.append({"template": template, "action": action, "qty": qty})

            if st.button("Use Custom Strategy", key="use_custom"):
                custom_config = {
                    "name": custom_name,
                    "conditions": conditions,
                    "condition_logic": cond_logic,
                    "legs": legs,
                }
                st.session_state["custom_strategy"] = custom_config
                st.success(f"Custom strategy '{custom_name}' configured!")

        st.divider()

        # Backtest configuration
        st.subheader("Backtest Configuration")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bt_symbol = st.selectbox("Symbol", list(INDEX_CONFIG.keys()) + ["RELIANCE", "INFY", "HDFCBANK"], key="bt_symbol")
        with bc2:
            from datetime import date, timedelta
            bt_start = st.date_input("Start Date", date.today() - timedelta(days=365), key="bt_start")
        with bc3:
            bt_end = st.date_input("End Date", date.today(), key="bt_end")

        bc4, bc5, bc6 = st.columns(3)
        with bc4:
            bt_capital = st.number_input("Initial Capital (INR)", 100_000, 50_000_000, 500_000, step=100_000, key="bt_capital")
        with bc5:
            bt_dte_min = st.number_input("Min DTE", 0, 90, 7, key="bt_dte_min")
            bt_dte_max = st.number_input("Max DTE", 1, 180, 45, key="bt_dte_max")
        with bc6:
            bt_sl = st.number_input("Stop Loss (%)", 0, 200, 50, key="bt_sl")
            bt_target = st.number_input("Target (%)", 0, 500, 50, key="bt_target")

        st.divider()

        # ── Data Management section ────────────────────────────────────
        with st.expander("Data Management", expanded=False):
            try:
                from data_downloader import get_available_date_range
                from data_cleaner import get_cache_stats, cleanup_date_range
                from config import BHAVCOPY_CACHE_DIR

                cache_stats = get_cache_stats()
                dm1, dm2, dm3 = st.columns(3)
                dm1.metric("Cached Files", cache_stats["total_files"])
                dm2.metric("Cache Size", f"{cache_stats['total_size_mb']} MB")
                earliest, latest = get_available_date_range()
                dm3.metric("Date Range",
                           f"{earliest} — {latest}" if earliest else "No data")

                st.markdown("**Delete cached data for a date range:**")
                from datetime import date as _date, timedelta as _td
                cl1, cl2, cl3 = st.columns([2, 2, 1])
                with cl1:
                    clean_start = st.date_input(
                        "From", _date.today() - _td(days=365), key="clean_start")
                with cl2:
                    clean_end = st.date_input(
                        "To", _date.today(), key="clean_end")
                with cl3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    do_clean = st.button("Delete Data", key="cleanup_btn", type="secondary")

                if do_clean:
                    result = cleanup_date_range(clean_start, clean_end)
                    if result["deleted"]:
                        st.success(
                            f"Removed {len(result['deleted'])} file(s) "
                            f"from {clean_start} to {clean_end}."
                        )
                        with st.expander("Deleted files"):
                            for f in result["deleted"]:
                                st.text(f)
                    else:
                        st.info("No cached files found in that date range.")
                    if result["errors"]:
                        for err in result["errors"]:
                            st.warning(err)

                st.caption(f"Data dir: `{BHAVCOPY_CACHE_DIR}`")
            except Exception as e:
                st.caption(f"Data management unavailable: {e}")

        st.divider()

        if st.button("Run Backtest", type="primary", key="run_bt"):
            # Show strategy configuration summary
            st.subheader("Strategy Configuration Summary")
            use_custom = st.session_state.get("custom_strategy")

            if use_custom:
                st.json(use_custom)
                st.success(f"Custom strategy: {use_custom['name']}")
            else:
                strat_info = filtered_strats.get(sel_strategy, {})
                st.markdown(f"**Strategy:** {sel_strategy}")
                st.markdown(f"**Category:** {strat_info.get('category', 'N/A')}")
                st.markdown(f"**Description:** {strat_info.get('description', 'N/A')}")

            st.markdown(f"**Symbol:** {bt_symbol}")
            st.markdown(f"**Period:** {bt_start} to {bt_end}")
            st.markdown(f"**Capital:** INR {bt_capital:,.0f}")
            st.markdown(f"**DTE Range:** {bt_dte_min} - {bt_dte_max}")
            st.markdown(f"**Stop Loss:** {bt_sl}% | **Target:** {bt_target}%")

            # ── Step 1: Auto-download data ──────────────────────────────
            st.divider()
            st.subheader("Acquiring Data")
            download_status = st.empty()
            download_progress = st.empty()
            download_log = st.container()

            try:
                from data_downloader import download_bhavcopies
                from datetime import date as _date

                download_status.info("Checking for cached data and downloading missing dates...")
                progress_messages = []

                def _progress_cb(msg: str):
                    progress_messages.append(msg)
                    download_progress.text(msg)

                dl_result = download_bhavcopies(
                    start=bt_start, end=bt_end, progress_cb=_progress_cb,
                )

                # Show download results
                download_progress.empty()
                if dl_result.total_downloaded > 0:
                    download_status.success(
                        f"Downloaded {dl_result.total_downloaded} new bhavcopy file(s). "
                        f"({dl_result.already_had} were already cached.)"
                    )
                elif dl_result.already_had > 0:
                    download_status.success(
                        f"All {dl_result.already_had} trading days already cached."
                    )
                else:
                    download_status.warning(
                        "Could not download data from any source. See details below."
                    )

                # Show source-by-source log
                with download_log.expander("Download details", expanded=False):
                    for msg in dl_result.messages:
                        st.text(msg)
                    for src, count in dl_result.source_results.items():
                        st.text(f"  {src}: {count} files")

            except Exception as dl_exc:
                download_status.warning(f"Auto-download unavailable: {dl_exc}")
                download_progress.empty()

            # ── Step 2: Load data and run backtest ──────────────────────
            data_found = False
            try:
                from backtester.data_adapter import load_multiple_bhavcopies, build_backtest_dataset
                from backtester.engine import BacktestEngine, BacktestConfig
                from config import BHAVCOPY_CACHE_DIR, DATA_DIR
                from data_downloader import touch_access
                from data_cleaner import mark_files_accessed
                from pathlib import Path

                DATA_DIR.mkdir(parents=True, exist_ok=True)
                bhavcopy_dir = BHAVCOPY_CACHE_DIR
                if bhavcopy_dir.exists():
                    csv_files = sorted(bhavcopy_dir.glob("*.csv")) + sorted(bhavcopy_dir.glob("*.zip"))
                    if csv_files:
                        with st.spinner("Loading bhavcopy data and running backtest..."):
                            options_df = load_multiple_bhavcopies(csv_files, symbol_filter=bt_symbol)
                            if options_df is not None and not options_df.empty:
                                data_found = True
                                # Mark files as accessed so they aren't cleaned up
                                mark_files_accessed(csv_files)

                                # Get lot size
                                lot = INDEX_CONFIG.get(bt_symbol, {}).get("lot_size", 25)

                                config = BacktestConfig(
                                    symbol=bt_symbol, lot_size=lot,
                                    start_date=bt_start, end_date=bt_end,
                                    initial_capital=float(bt_capital),
                                    dte_min=bt_dte_min, dte_max=bt_dte_max,
                                    stop_loss_pct=float(bt_sl) if bt_sl > 0 else None,
                                    target_pct=float(bt_target) if bt_target > 0 else None,
                                )
                                engine = BacktestEngine(config, options_df)

                                # Get strategy function
                                if use_custom:
                                    strat_fn = CustomStrategy(use_custom)
                                else:
                                    strat_info = filtered_strats.get(sel_strategy, {})
                                    strat_fn = strat_info.get("fn")

                                if strat_fn:
                                    report = engine.run(strat_fn)

                                    # Display results
                                    st.subheader("Backtest Results")
                                    r1, r2, r3, r4 = st.columns(4)
                                    r1.metric("Total Trades", report.total_trades)
                                    r2.metric("Win Rate", f"{report.win_rate}%")
                                    r3.metric("Total P&L", f"INR {report.total_pnl:,.0f}")
                                    r4.metric("Return", f"{report.return_pct}%")

                                    r5, r6, r7, r8 = st.columns(4)
                                    r5.metric("Avg P&L/Trade", f"INR {report.avg_pnl_per_trade:,.0f}")
                                    r6.metric("Profit Factor", report.profit_factor)
                                    r7.metric("Max Drawdown", f"INR {report.max_drawdown:,.0f}")
                                    r8.metric("Sharpe Ratio", report.sharpe_ratio)

                                    # Equity curve
                                    if not report.equity_curve.empty:
                                        eq_fig = go.Figure()
                                        eq_fig.add_trace(go.Scatter(
                                            x=report.equity_curve["date"],
                                            y=report.equity_curve["equity"],
                                            mode="lines", name="Equity",
                                            line=dict(color="#42a5f5"),
                                        ))
                                        eq_fig.add_hline(y=config.initial_capital, line_dash="dash", line_color="white", opacity=0.5)
                                        eq_fig.update_layout(
                                            title="Equity Curve", height=400, template="plotly_dark",
                                            yaxis_title="Capital (INR)",
                                        )
                                        st.plotly_chart(eq_fig, width="stretch")

                                    # Trade log
                                    if report.trades:
                                        st.subheader("Trade Log")
                                        trade_rows = []
                                        for t in report.trades:
                                            trade_rows.append({
                                                "Entry": str(t.entry_date),
                                                "Exit": str(t.exit_date),
                                                "Strategy": t.strategy_name,
                                                "Entry Premium": t.entry_premium,
                                                "Exit Premium": t.exit_premium,
                                                "P&L": t.pnl,
                                                "P&L %": t.pnl_pct,
                                                "Total P&L": t.total_pnl,
                                                "DTE": t.dte_at_entry,
                                            })
                                        trade_df = pd.DataFrame(trade_rows)
                                        st.dataframe(trade_df, width="stretch", hide_index=True)
            except ImportError as e:
                st.caption(f"Backtest engine: {e}")
            except Exception as e:
                st.error(f"Backtest error: {e}")

            # ── No data found — show detailed guidance ────────────────
            if not data_found:
                st.divider()
                st.error(
                    "No historical F&O bhavcopy data could be obtained for backtesting. "
                    "All automatic download sources were attempted."
                )
                st.markdown(
                    """
### Manual data acquisition

The automatic downloader tried all available sources but could not obtain data.
This can happen if NSE is blocking requests, the date range has no data, or
network issues occurred. You can manually acquire the data:

---

**Option 1 — Download from NSE website**

1. Visit [NSE Historical Data](https://www.nseindia.com/all-reports-derivatives#702)
2. Select **"F&O Bhavcopy"** and the date range
3. Download the ZIP files and place them in:
   ```
   ~/.indian-options/bhavcopy/
   ```

---

**Option 2 — Use `jugaad-data` package**

```bash
pip install jugaad-data
```

```python
from jugaad_data.nse import bhavcopy_fo_save
from datetime import date, timedelta
import os

save_dir = os.path.expanduser("~/.indian-options/bhavcopy")
os.makedirs(save_dir, exist_ok=True)

start = date.today() - timedelta(days=30)
d = start
while d <= date.today():
    try:
        bhavcopy_fo_save(d, save_dir)
        print(f"Downloaded: {d}")
    except Exception:
        pass  # skip holidays
    d += timedelta(days=1)
```

---

**Expected CSV columns:**
```
INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
OPEN_INT, CHG_IN_OI, TIMESTAMP
```
                    """,
                    unsafe_allow_html=True,
                )

    except ImportError as e:
        st.info(f"Backtester modules loading: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 – MARKET NEWS
# ══════════════════════════════════════════════════════════════════════════

with tab_news:
    st.header(f"{display_name} – Market News & Impact")

    with st.spinner("Fetching market news..."):
        news_df, news_ts = _news(selected_index)

    if news_df is not None and not news_df.empty:
        # Filters row
        nf1, nf2 = st.columns(2)
        with nf1:
            categories = ["All"] + sorted(news_df["Category"].unique().tolist())
            sel_cat = st.selectbox("Filter by Category", categories, index=0, key="news_cat")
        with nf2:
            sent_opts = ["All", "Positive", "Negative", "Neutral"]
            sel_sent = st.selectbox("Filter by Sentiment", sent_opts, index=0, key="news_sent")

        filtered = news_df.copy()
        if sel_cat != "All":
            filtered = filtered[filtered["Category"] == sel_cat]
        if sel_sent != "All":
            filtered = filtered[filtered["Sentiment"] == sel_sent]

        # Counts
        live_count = int(filtered["IsLive"].sum())
        total = len(filtered)
        pos_count = int((filtered["Sentiment"] == "Positive").sum())
        neg_count = int((filtered["Sentiment"] == "Negative").sum())
        neu_count = int((filtered["Sentiment"] == "Neutral").sum())

        ns0, ns1, ns2, ns3, ns4 = st.columns(5)
        ns0.metric("LIVE", live_count)
        ns1.metric("Total", total)
        ns2.metric("Positive", pos_count)
        ns3.metric("Negative", neg_count)
        ns4.metric("Neutral", neu_count)

        # Sentiment distribution chart
        if total > 0:
            sent_fig = go.Figure(data=[
                go.Pie(
                    labels=["Positive", "Negative", "Neutral"],
                    values=[pos_count, neg_count, neu_count],
                    marker=dict(colors=["#66bb6a", "#ef5350", "#fdd835"]),
                    hole=0.4,
                )
            ])
            sent_fig.update_layout(
                title="Sentiment Distribution",
                height=300, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark",
            )
            st.plotly_chart(sent_fig, width="stretch")

        # News cards – newest first (already sorted by PublishedDT desc)
        for _, row in filtered.iterrows():
            sent_color = {"Positive": "#66bb6a", "Negative": "#ef5350"}.get(row["Sentiment"], "#fdd835")
            cat_color = {
                "Options / F&O": "#42a5f5",
                "Market Outlook": "#ab47bc",
                "Macro / Policy": "#ff7043",
                "Sector / Stock": "#26a69a",
                "Technical": "#5c6bc0",
            }.get(row["Category"], "#78909c")

            live_badge = (
                '<span style="background:#e53935; color:white; padding:2px 8px; '
                'border-radius:10px; font-size:0.7rem; font-weight:700; '
                'animation:pulse 1.5s infinite;">LIVE</span> '
                if row.get("IsLive") else ""
            )
            time_ago_str = row.get("TimeAgo", "")

            st.markdown(
                f'<div style="border-left: 4px solid {sent_color}; padding: 8px 14px; margin: 6px 0; '
                f'background: #1e1e2f; border-radius: 4px;">'
                f'{live_badge}'
                f'<span style="background:{cat_color}; color:white; padding:2px 8px; border-radius:10px; '
                f'font-size:0.75rem; font-weight:600;">{row["Category"]}</span> &nbsp; '
                f'<span style="color:{sent_color}; font-weight:700; font-size:0.8rem;">'
                f'{row["Sentiment"].upper()}</span> &nbsp; '
                f'<span style="color:#aaa; font-size:0.75rem; font-weight:600;">{time_ago_str}</span>'
                f'<br/><a href="{row["Link"]}" target="_blank" style="color:#e0e0e0; '
                f'text-decoration:none; font-weight:600;">{row["Title"]}</a>'
                + (f'<br/><span style="color:#aaa; font-size:0.85rem;">{row["Summary"]}</span>' if row["Summary"] else "")
                + f'<br/><span style="color:#666; font-size:0.7rem;">Source: {row["Source"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(freshness(news_ts), unsafe_allow_html=True)
    else:
        st.info("No news articles available at the moment. Try refreshing.")


# ── Footer ─────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data sourced from NSE India & yfinance (free, no API key required). "
    "This tool is for informational purposes only and does not constitute trading advice. "
    "Always do your own analysis before placing trades."
)
