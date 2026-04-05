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

logging.basicConfig(level=logging.WARNING)

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
    return f'<span class="freshness">Updated: {dt.strftime("%H:%M:%S IST")}</span>'


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
    refresh = st.button("Refresh All Data", type="primary", use_container_width=True)
    st.divider()
    phase = market_phase()
    st.markdown(phase_badge(phase), unsafe_allow_html=True)
    st.caption(f"IST: {datetime.now(IST).strftime('%H:%M:%S')}")

display_name = INDEX_CONFIG[selected_index]["display_name"]
st.title(f"Pre-Trade Checklist  –  {display_name}")

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
# SECTION 1 – GLOBAL CUES & MACRO
# ══════════════════════════════════════════════════════════════════════════

st.header("1. Global Cues & Macro")
col_global, col_vix, col_fii = st.columns([2, 1, 1.5])

with col_global:
    with st.spinner("Fetching global indices..."):
        global_df, global_ts = _global_cues()
    if global_df is not None and not global_df.empty:
        styled = global_df.style.map(colour_change, subset=["Change %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
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
        st.dataframe(fii_dii_df, use_container_width=True, hide_index=True)
        st.markdown(freshness(fii_ts), unsafe_allow_html=True)
    else:
        st.caption("FII/DII data unavailable (NSE may restrict automated access).")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 – SPOT & TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

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

    t1, t2 = st.columns(2)

    with t1:
        st.subheader("Moving Averages")
        ema_rows = []
        for period, val in technicals["emas"].items():
            pos = "Above" if technicals["spot"] > val else "Below"
            ema_rows.append({"EMA": f"EMA {period}", "Value": val, "Spot vs EMA": pos})
        if ema_rows:
            ema_df = pd.DataFrame(ema_rows)
            def _ema_colour(val):
                return "color: #4caf50; font-weight:700" if val == "Above" else "color: #ef5350; font-weight:700"
            st.dataframe(
                ema_df.style.map(_ema_colour, subset=["Spot vs EMA"]),
                use_container_width=True, hide_index=True,
            )

        st.subheader("Momentum")
        mc1, mc2 = st.columns(2)
        mc1.metric("RSI (14)", technicals["rsi"] if technicals["rsi"] else "N/A")
        macd_d = technicals["macd"]
        mc2.metric("MACD Hist", macd_d["histogram"] if macd_d else "N/A")

    with t2:
        st.subheader("Pivot Points (Classic)")
        pivots = technicals.get("pivots")
        if pivots:
            p_df = pd.DataFrame([pivots]).T.reset_index()
            p_df.columns = ["Level", "Price"]
            st.dataframe(p_df, use_container_width=True, hide_index=True)

        st.subheader("Fibonacci Retracement")
        fib = technicals.get("fibonacci")
        if fib:
            f_df = pd.DataFrame([fib]).T.reset_index()
            f_df.columns = ["Level", "Price"]
            st.dataframe(f_df, use_container_width=True, hide_index=True)

    st.markdown(freshness(hist_ts), unsafe_allow_html=True)
else:
    st.warning("Historical data unavailable – technical analysis skipped.")

# Intraday candlestick chart
if intra_df is not None and not intra_df.empty:
    st.subheader(f"Intraday Chart ({timeframe})")
    candle_fig = go.Figure(
        data=[
            go.Candlestick(
                x=intra_df.index,
                open=intra_df["open"], high=intra_df["high"],
                low=intra_df["low"], close=intra_df["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            )
        ]
    )
    candle_fig.update_layout(
        xaxis_rangeslider_visible=False, height=400,
        margin=dict(l=40, r=20, t=30, b=30), template="plotly_dark",
    )
    st.plotly_chart(candle_fig, use_container_width=True)
    st.markdown(freshness(intra_ts), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 – OPTIONS CHAIN ANALYSIS  (NSE)
# ══════════════════════════════════════════════════════════════════════════

st.header(f"3. {display_name} – Options Chain Analysis")

with st.spinner("Fetching NSE option chain (may take a moment)..."):
    nse_data, oc_ts = _nse_option_chain(selected_index)

if not nse_data:
    st.warning(
        "Could not fetch option chain from NSE. "
        "NSE may be blocking automated requests — try refreshing in a few seconds."
    )
    # Graceful degradation: still show checklist with what we have
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
    selected_expiry = st.selectbox("Expiry Date", expiry_list, index=0) if expiry_list else None

    options_result = run_options_analysis(nse_data, selected_expiry)
    spot_for_oc = options_result["spot"] or ltp
    chain_df: pd.DataFrame = options_result["chain_df"]

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
        st.plotly_chart(oi_fig, use_container_width=True)

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
        st.plotly_chart(iv_fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 – PRE-TRADE CHECKLIST & OVERALL BIAS
# ══════════════════════════════════════════════════════════════════════════

st.header("4. Pre-Trade Checklist Summary")

checklist = generate_checklist(
    technicals=technicals if technicals else {"spot": ltp, "emas": {}, "rsi": None, "macd": None},
    options_data=options_result,
    vix=vix_value,
    global_df=global_df if global_df is not None else pd.DataFrame(),
    fii_dii_df=fii_dii_df,
)

overall_label, overall_pct = compute_overall_bias(checklist)
st.markdown(bias_box_html(overall_label, overall_pct), unsafe_allow_html=True)

# Checklist table
header_cols = st.columns([3, 3, 2, 2])
header_cols[0].markdown("**Indicator**")
header_cols[1].markdown("**Value**")
header_cols[2].markdown("**Signal**")
header_cols[3].markdown("**Weight**")
st.divider()
for item in checklist:
    cols = st.columns([3, 3, 2, 2])
    cols[0].write(item["indicator"])
    cols[1].write(str(item["value"]))
    cols[2].markdown(signal_badge(item["signal"]), unsafe_allow_html=True)
    cols[3].write(f'{item["weight"]:.1f}')

# Bias gauge
st.subheader("Bias Gauge")
gauge_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=overall_pct,
        title={"text": "Market Bias Score"},
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
gauge_fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=30), template="plotly_dark")
st.plotly_chart(gauge_fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 – MARKET NEWS & IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

st.header(f"5. {display_name} – Market News & Impact")

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
    # Already sorted newest-first via PublishedDT in the fetcher

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
        st.plotly_chart(sent_fig, use_container_width=True)

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
