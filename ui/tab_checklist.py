"""Pre-Trade Checklist tab – Section 1-4 of the dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import INDEX_CONFIG
from analysis import (
    run_technical_analysis,
    run_options_analysis,
    get_expiry_dates,
    generate_checklist,
    compute_overall_bias,
    _ema,
    _rsi,
    _macd,
    BULLISH,
    BEARISH,
    NEUTRAL,
)
from ui.components import signal_badge, bias_box_html, freshness, colour_change, _build_checklist_html
from ui.cache import _global_cues, _india_vix, _spot_data, _historical, _intraday, _nse_option_chain, _fii_dii


def render(selected_index: str, display_name: str, timeframe: str) -> None:
    """Render the Pre-Trade Checklist tab content.

    Stores shared data in ``st.session_state`` so other tabs can access it:
    - ``_checklist_nse_data``
    - ``_checklist_ltp``
    - ``_checklist_options_result``
    """

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

        # Build the main chart with subplots: candlestick + volume + RSI + MACD
        tech_fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.50, 0.12, 0.19, 0.19],
            subplot_titles=("", "Volume", "RSI (14)", "MACD"),
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
        ema_colours = {9: "#fdd835", 20: "#42a5f5", 50: "#ab47bc", 200: "#ef6c00"}
        for period in [9, 20, 50, 200]:
            ema_series = _ema(chart_df["close"], length=period)
            if not ema_series.empty and not ema_series.dropna().empty:
                tech_fig.add_trace(
                    go.Scatter(
                        x=chart_df.index, y=ema_series,
                        mode="lines", name=f"EMA {period}",
                        line=dict(color=ema_colours.get(period, "#ffffff"), width=1.2),
                    ),
                    row=1, col=1,
                )

        # 2b) VWAP (intraday only)
        if chart_source == "Intraday" and "volume" in chart_df.columns:
            cum_vol = chart_df["volume"].cumsum()
            cum_vwap = (chart_df["close"] * chart_df["volume"]).cumsum()
            vwap = cum_vwap / cum_vol.replace(0, float("nan"))
            if not vwap.dropna().empty:
                tech_fig.add_trace(
                    go.Scatter(
                        x=chart_df.index, y=vwap,
                        mode="lines", name="VWAP",
                        line=dict(color="#00e5ff", width=1.5, dash="dot"),
                    ),
                    row=1, col=1,
                )

        # 2c) Volume bars
        if "volume" in chart_df.columns:
            vol_colors = [
                "#66bb6a" if chart_df["close"].iloc[i] >= chart_df["open"].iloc[i]
                else "#ef5350"
                for i in range(len(chart_df))
            ]
            tech_fig.add_trace(
                go.Bar(
                    x=chart_df.index, y=chart_df["volume"],
                    name="Volume", marker_color=vol_colors, opacity=0.6,
                    showlegend=False,
                ),
                row=2, col=1,
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
        rsi_series = _rsi(chart_df["close"], length=14)
        if rsi_series is not None and not rsi_series.dropna().empty:
            tech_fig.add_trace(
                go.Scatter(x=chart_df.index, y=rsi_series, mode="lines", name="RSI",
                           line=dict(color="#ab47bc", width=1.5)),
                row=3, col=1,
            )
            tech_fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.5, row=3, col=1)
            tech_fig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", opacity=0.5, row=3, col=1)
            tech_fig.add_hrect(y0=30, y1=70, fillcolor="#fdd835", opacity=0.05, row=3, col=1)

        # 6) MACD subplot
        macd_result = _macd(chart_df["close"], fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.dropna().empty:
            tech_fig.add_trace(
                go.Scatter(x=chart_df.index, y=macd_result["macd"], mode="lines",
                           name="MACD", line=dict(color="#42a5f5", width=1.2)),
                row=4, col=1,
            )
            tech_fig.add_trace(
                go.Scatter(x=chart_df.index, y=macd_result["signal"], mode="lines",
                           name="Signal", line=dict(color="#ef6c00", width=1.2)),
                row=4, col=1,
            )
            hist_vals = macd_result["histogram"]
            colours = ["#66bb6a" if v >= 0 else "#ef5350" for v in hist_vals]
            tech_fig.add_trace(
                go.Bar(x=chart_df.index, y=hist_vals, name="Histogram",
                       marker_color=colours, opacity=0.6),
                row=4, col=1,
            )

        # Layout
        tech_fig.update_layout(
            title=chart_title,
            height=800,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font_size=10),
            hovermode="x unified",
        )
        tech_fig.update_yaxes(title_text="Price", row=1, col=1)
        tech_fig.update_yaxes(title_text="Vol", row=2, col=1)
        tech_fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        tech_fig.update_yaxes(title_text="MACD", row=4, col=1)

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

    # ── Store shared data for other tabs ──────────────────────────────
    st.session_state["_checklist_nse_data"] = nse_data
    st.session_state["_checklist_ltp"] = ltp
    st.session_state["_checklist_options_result"] = options_result
