"""
Market Advisor tab — day classification, strategy recommendations,
and comprehensive JSON strategy import.

Brings together live market data (VIX, PCR, technicals) to classify
the trading day and recommend appropriate strategies. Also provides
a full JSON import for the comprehensive strategy format.
"""

import json
import logging

import streamlit as st
import pandas as pd

from config import IST, INDEX_CONFIG, INDIA_VIX_TICKER

logger = logging.getLogger(__name__)


# ── Section 1: Market Regime Dashboard ────────────────────────────────────

def _render_regime_dashboard(selected_index: str, display_name: str, timeframe: str):
    """Render the live market regime classification dashboard."""
    st.subheader("Today's Market Regime")
    st.caption(
        "Analyses VIX, ADX, gap, CPR, PCR and expiry status to classify the day "
        "and recommend strategies. Data refreshes with the sidebar refresh button."
    )

    try:
        from core.market_regime import (
            VIXRegime, PCRRegime, GapAnalysis, CPRAnalysis,
            classify_day, recommend_strategies, MarketTimeContext, RiskContext,
        )
    except ImportError as e:
        st.error(f"Market regime module not available: {e}")
        return

    # ── Fetch live data ──
    vix_value = 0.0
    pcr_value = 0.0
    adx_value = None
    gap_analysis = None
    cpr_analysis = None
    spot = 0.0

    # VIX
    try:
        from services.data_fetcher import fetch_india_vix
        vix_data = fetch_india_vix()
        if vix_data and "current" in vix_data:
            vix_value = float(vix_data["current"])
        elif vix_data and "close" in vix_data:
            vix_value = float(vix_data["close"])
    except Exception as e:
        logger.warning("Could not fetch VIX: %s", e)

    # Option chain for PCR
    try:
        from services.data_fetcher import fetch_option_chain
        from analysis.options import parse_nse_option_chain, compute_pcr, get_underlying_value
        raw_chain = fetch_option_chain(selected_index)
        if raw_chain:
            chain_df, meta = parse_nse_option_chain(raw_chain)
            spot = get_underlying_value(raw_chain) or 0.0
            pcr_result = compute_pcr(chain_df)
            if pcr_result and "pcr" in pcr_result:
                pcr_value = float(pcr_result["pcr"])
    except Exception as e:
        logger.warning("Could not fetch option chain for PCR: %s", e)

    # Technical data for ADX, gap, CPR
    try:
        from services.data_fetcher import fetch_historical_data
        idx_cfg = INDEX_CONFIG.get(selected_index, {})
        ticker = idx_cfg.get("yf_ticker", "^NSEI")
        hist = fetch_historical_data(ticker, period="1mo", interval="1d")
        if hist is not None and len(hist) >= 15:
            # ADX calculation
            high = hist["High"]
            low = hist["Low"]
            close = hist["Close"]
            import numpy as np
            period = 14
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            hl = high - low
            hc = (high - close.shift(1)).abs()
            lc = (low - close.shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
            adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
            adx_val = adx_series.iloc[-1]
            if not pd.isna(adx_val):
                adx_value = float(adx_val)

            # Gap analysis
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                curr_open = float(hist["Open"].iloc[-1])
                if spot <= 0:
                    spot = float(hist["Close"].iloc[-1])
                gap_analysis = GapAnalysis.classify(prev_close, curr_open)

            # CPR analysis
            if len(hist) >= 2:
                prev_row = hist.iloc[-2]
                cpr_analysis = CPRAnalysis.from_prev_day(
                    float(prev_row["High"]), float(prev_row["Low"]),
                    float(prev_row["Close"]), spot,
                )
    except Exception as e:
        logger.warning("Could not fetch historical data: %s", e)

    # ── Classify the day ──
    day_class = classify_day(
        vix_value=vix_value,
        pcr_value=pcr_value,
        adx_value=adx_value,
        gap_analysis=gap_analysis,
        cpr_analysis=cpr_analysis,
    )

    # ── Display regime ──
    type_colors = {
        "trending": "#42a5f5",
        "range_bound": "#66bb6a",
        "volatile": "#ef5350",
        "expiry": "#ffa726",
    }
    type_icons = {
        "trending": "TRENDING",
        "range_bound": "RANGE-BOUND",
        "volatile": "VOLATILE",
        "expiry": "EXPIRY DAY",
    }
    color = type_colors.get(day_class.day_type, "#888")
    icon = type_icons.get(day_class.day_type, day_class.day_type.upper())

    st.markdown(
        f'<div style="background:linear-gradient(135deg, {color}22, {color}44); '
        f'border:2px solid {color}; border-radius:12px; padding:20px; text-align:center; margin:10px 0;">'
        f'<h2 style="color:{color}; margin:0;">{icon}</h2>'
        f'<p style="color:#ccc; margin:5px 0;">{day_class.summary}</p>'
        f'<p style="color:#888; font-size:0.8rem;">Position size: {day_class.position_size_multiplier:.0%} of normal</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Factor breakdown
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        vix_color = {"low": "green", "normal": "blue", "high": "orange", "extreme": "red"}.get(day_class.vix.level, "gray")
        st.metric("India VIX", f"{vix_value:.1f}", delta=day_class.vix.level.upper())
    with m2:
        pcr_color = "green" if day_class.pcr.bias == "bullish" else ("red" if day_class.pcr.bias == "bearish" else "gray")
        st.metric("PCR", f"{pcr_value:.2f}", delta=day_class.pcr.bias.upper())
    with m3:
        adx_str = f"{adx_value:.0f}" if adx_value else "N/A"
        adx_delta = "Strong" if adx_value and adx_value > 25 else ("Weak" if adx_value and adx_value < 20 else "Moderate")
        st.metric("ADX", adx_str, delta=adx_delta)
    with m4:
        gap_str = f"{gap_analysis.gap_pct:+.1f}%" if gap_analysis else "N/A"
        st.metric("Gap", gap_str, delta=gap_analysis.direction if gap_analysis else "N/A")

    # Factors list
    with st.expander("Classification Factors", expanded=False):
        for factor in day_class.factors:
            st.markdown(f"- {factor}")

    # CPR levels
    if cpr_analysis:
        with st.expander("CPR & Pivot Levels", expanded=False):
            lvl1, lvl2, lvl3 = st.columns(3)
            lvl1.metric("Pivot", f"{cpr_analysis.pivot:,.0f}")
            lvl2.metric("R1 / S1", f"{cpr_analysis.r1:,.0f} / {cpr_analysis.s1:,.0f}")
            lvl3.metric("R2 / S2", f"{cpr_analysis.r2:,.0f} / {cpr_analysis.s2:,.0f}")
            narrow_str = "Narrow (trending day signal)" if cpr_analysis.is_narrow else "Wide (range-bound signal)"
            st.caption(f"CPR Width: {cpr_analysis.cpr_width_pct:.3f}% — {narrow_str}")

    # ── Time context ──
    time_ctx = MarketTimeContext.now()
    phase_colors = {
        "pre_market": "#78909c",
        "opening_avoid": "#ef5350",
        "first_trade": "#66bb6a",
        "morning_prime": "#42a5f5",
        "lunch_lull": "#ffa726",
        "afternoon": "#42a5f5",
        "exit_zone": "#ef5350",
        "closed": "#78909c",
    }
    pc = phase_colors.get(time_ctx.phase, "#888")
    st.markdown(
        f'<div style="background:{pc}22; border-left:4px solid {pc}; padding:10px 14px; '
        f'border-radius:4px; margin:10px 0;">'
        f'<strong style="color:{pc};">{time_ctx.phase.replace("_", " ").title()}</strong>'
        f'{"  |  Can enter" if time_ctx.can_enter else "  |  No new entries"}'
        f' | {time_ctx.guidance}'
        f'</div>',
        unsafe_allow_html=True,
    )

    return day_class


# ── Section 2: Strategy Recommendations ───────────────────────────────────

def _render_recommendations(day_class):
    """Render strategy recommendations based on day classification."""
    st.subheader("Recommended Strategies")
    st.caption(
        "Based on today's regime, these strategies have the highest probability "
        "of success. Click 'Load' to use one in the backtester."
    )

    try:
        from core.market_regime import recommend_strategies
        from backtester.custom_strategy import describe_strategy
    except ImportError:
        st.warning("Strategy recommendation module not available")
        return

    recommendations = recommend_strategies(day_class)
    if not recommendations:
        st.info("No specific recommendations for the current market regime.")
        return

    for i, rec in enumerate(recommendations):
        risk_colors = {"low": "#66bb6a", "medium": "#ffa726", "high": "#ef5350"}
        rc = risk_colors.get(rec.risk_level, "#888")

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f'<div style="background:#1a1a2e; padding:12px 16px; border-radius:8px; '
                f'margin:6px 0; border-left:4px solid {rc};">'
                f'<strong>{rec.name}</strong> '
                f'<span style="color:{rc}; font-size:0.75rem; background:{rc}22; '
                f'padding:2px 8px; border-radius:4px; margin-left:8px;">'
                f'{rec.risk_level.upper()} RISK</span>'
                f'<br/><span style="color:#aaa;">{rec.description}</span>'
                f'<br/><span style="color:#888; font-size:0.8rem;">'
                f'Why: {rec.suitability} | Window: {rec.entry_window}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            if rec.config and st.button("Load", key=f"rec_load_{i}"):
                st.session_state["custom_strategy"] = rec.config
                st.success(f"Loaded '{rec.name}' — switch to Strategy Backtester tab to run it.")

    # Risk context
    st.markdown("---")
    st.markdown("#### Risk Parameters for Today")
    from core.market_regime import RiskContext
    risk = RiskContext(
        capital=500000,
        position_size_multiplier=day_class.position_size_multiplier,
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Max Risk/Trade", f"INR {risk.max_risk_per_trade:,.0f}")
    r2.metric("Max Daily Loss", f"INR {risk.max_daily_loss:,.0f}")
    r3.metric("Position Sizing", f"{day_class.position_size_multiplier:.0%} of normal")


# ── Section 3: Comprehensive JSON Import ──────────────────────────────────

def _render_json_import():
    """Render the comprehensive JSON strategy import section."""
    st.subheader("Import Full Strategy Config (JSON)")
    st.caption(
        "Import a comprehensive strategy JSON with entry/exit rules, risk management, "
        "day classification, Greeks guidance, and more. The system will parse it and "
        "convert each strategy to a format our backtester can execute."
    )

    try:
        from core.strategy_schema import (
            load_strategy_from_file, load_strategy_from_json_string,
            summarize_strategy_config, validate_strategy_json,
        )
    except ImportError as e:
        st.warning(f"Strategy schema module not available: {e}")
        return

    input_method = st.radio(
        "Input method", ["Upload JSON file", "Paste JSON"],
        horizontal=True, key="json_input_method",
    )

    full_config = None
    validation = None

    if input_method == "Upload JSON file":
        uploaded = st.file_uploader(
            "Upload strategy JSON",
            type=["json"],
            key="full_json_upload",
            help="Upload a comprehensive strategy JSON file (like the Indian Options Intraday Master Strategy)",
        )
        if uploaded:
            content = uploaded.read()
            full_config, validation = load_strategy_from_file(content)
    else:
        json_text = st.text_area(
            "Paste JSON here",
            height=300,
            key="full_json_paste",
            placeholder='{\n  "strategy_meta": { "name": "My Strategy", ... },\n  "strategies": { ... },\n  ...\n}',
        )
        if json_text.strip():
            full_config, validation = load_strategy_from_json_string(json_text)

    if validation:
        # Show validation results
        if validation.valid:
            st.success(
                f"Valid JSON — {validation.strategies_found} strategies found, "
                f"{validation.conditions_mapped} conditions mapped"
            )
        else:
            st.error("Invalid JSON — see errors below")
            for err in validation.errors:
                st.markdown(f"- {err}")

        if validation.warnings:
            with st.expander(f"Warnings ({len(validation.warnings)})", expanded=False):
                for w in validation.warnings:
                    st.markdown(f"- {w}")

    if full_config:
        # Show summary
        st.markdown("---")
        st.markdown("### Strategy Overview")
        summary = summarize_strategy_config(full_config)
        st.markdown(summary)

        # Show individual strategies with load buttons
        st.markdown("---")
        st.markdown("### Individual Strategies")
        st.caption(
            "Each strategy has been converted to our backtester format. "
            "Some parameters (trailing stops, adjustments, time windows) are noted "
            "but cannot be backtested with daily data."
        )

        for i, strat in enumerate(full_config.strategies):
            with st.expander(f"{strat.name} ({strat.strategy_type})", expanded=False):
                st.markdown(f"**{strat.description}**")

                if strat.best_conditions:
                    st.markdown(f"Best conditions: {', '.join(strat.best_conditions)}")

                if strat.entry_window:
                    st.markdown(f"Entry window: {strat.entry_window.start} - {strat.entry_window.end} IST")

                if strat.exit_rules:
                    er = strat.exit_rules
                    st.markdown(
                        f"Exit: Target {er.target_pct}% | SL {er.stop_loss_pct}% | "
                        f"Exit by {er.mandatory_exit_time}"
                    )
                    if er.trailing_stop:
                        st.markdown(f"Trailing stop: {er.trailing_stop}")
                    if er.partial_booking:
                        st.markdown(f"Partial booking: {er.partial_booking}")

                if strat.adjustment_rules:
                    ar = strat.adjustment_rules
                    st.markdown(f"Adjustments: {ar.trigger}")
                    for action in ar.actions:
                        st.markdown(f"  - {action}")

                if strat.position_sizing:
                    ps = strat.position_sizing
                    st.markdown(
                        f"Position sizing: Max {ps.max_lots} lots, "
                        f"{ps.max_capital_per_trade_pct}% capital/trade"
                    )

                if strat.legs:
                    st.markdown("**Legs:**")
                    for leg in strat.legs:
                        st.markdown(
                            f"  - {leg.get('action', '?')} {leg.get('type', '?')} "
                            f"@ {leg.get('strike', 'ATM')} x{leg.get('lots', 1)}"
                        )

                # Converted config
                if strat.custom_strategy_config:
                    st.markdown("**Converted to backtester format:**")
                    config = strat.custom_strategy_config
                    try:
                        from backtester.custom_strategy import describe_strategy
                        st.markdown(describe_strategy(config))
                    except Exception:
                        pass

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Load for Backtesting", key=f"load_imported_{i}"):
                            st.session_state["custom_strategy"] = config
                            st.success(f"Loaded '{strat.name}' — switch to Strategy Backtester tab.")
                    with col_b:
                        st.download_button(
                            "Export converted JSON",
                            data=json.dumps(config, indent=2, default=str),
                            file_name=f"{strat.name.replace(' ', '_').lower()}_converted.json",
                            mime="application/json",
                            key=f"export_imported_{i}",
                        )

                # Show limitations if relevant
                limitations = []
                if strat.entry_window:
                    limitations.append("Time-based entry windows (backtester uses daily data)")
                if strat.exit_rules and strat.exit_rules.trailing_stop:
                    limitations.append("Trailing stops (engine supports fixed SL/target only)")
                if strat.exit_rules and strat.exit_rules.partial_booking:
                    limitations.append("Partial profit booking (engine exits full position)")
                if strat.adjustment_rules and strat.adjustment_rules.actions:
                    limitations.append("Mid-trade adjustments (engine doesn't support leg modification)")
                if strat.strike_selection and "delta" in str(strat.strike_selection).lower():
                    limitations.append("Delta-based strike selection (no Greeks calculation)")
                if limitations:
                    st.warning(
                        "**Not backtestable with current engine:** " + "; ".join(limitations) +
                        ". These are noted for live trading reference."
                    )

        # Risk management summary
        if full_config.risk_management:
            st.markdown("---")
            st.markdown("### Risk Management Rules")
            rm = full_config.risk_management
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Daily Loss Limit", f"{rm.max_daily_loss_pct}% / INR {rm.max_daily_loss_inr:,.0f}")
            rc2.metric("Weekly Limit", f"{rm.max_weekly_loss_pct}%")
            rc3.metric("Monthly Drawdown", f"{rm.max_monthly_drawdown_pct}%")

            rc4, rc5, rc6 = st.columns(3)
            rc4.metric("Max Trades/Day", rm.max_trades_per_day)
            rc5.metric("Stop After Losses", f"{rm.stop_after_consecutive_losses} consecutive")
            rc6.metric("Max Concurrent", rm.max_concurrent_positions)

            st.caption(
                f"Per-trade: Max risk {rm.max_risk_per_trade_pct}% | "
                f"Max capital {rm.max_capital_per_trade_pct}% | "
                f"Slippage {rm.slippage_pct}% | "
                f"Round-trip cost {rm.total_round_trip_cost_pct}%"
            )
            if rm.no_averaging:
                st.caption("No averaging losing positions | No doubling down")

        # Greeks guidance
        if full_config.greeks_guidance:
            st.markdown("---")
            with st.expander("Greeks Guidance (Reference)", expanded=False):
                for greek, guidance in full_config.greeks_guidance.items():
                    st.markdown(f"**{greek.title()}**")
                    if isinstance(guidance, dict):
                        for k, v in guidance.items():
                            st.markdown(f"  - **{k.replace('_', ' ').title()}:** {v}")
                    else:
                        st.markdown(f"  {guidance}")

        # Day classification from imported config
        if full_config.day_classification:
            st.markdown("---")
            with st.expander("Imported Day Classification Rules", expanded=False):
                for day_type, rules in full_config.day_classification.items():
                    if isinstance(rules, dict):
                        st.markdown(f"**{day_type.replace('_', ' ').title()}**")
                        signs = rules.get("signs", [])
                        preferred = rules.get("preferred_strategies", [])
                        avoid = rules.get("avoid_strategies", [])
                        if signs:
                            st.markdown(f"  Signs: {', '.join(signs)}")
                        if preferred:
                            st.markdown(f"  Preferred: {', '.join(preferred)}")
                        if avoid:
                            st.markdown(f"  Avoid: {', '.join(avoid)}")

        # Store full config in session
        st.session_state["imported_full_config"] = full_config


# ── Section 4: Historical Date Analysis ───────────────────────────────────

def _render_historical_analysis(selected_index: str, display_name: str):
    """Render analysis for any historical date using real data."""
    st.subheader("Analyse Any Date")
    st.caption(
        "Pick any historical trading date to see the full analysis — "
        "real OHLC from yfinance, VIX, option chain from bhavcopy, "
        "all technical indicators, day classification, and strategy recommendations."
    )

    try:
        from core.historical_analyzer import analyze_date, get_available_analysis_dates
    except ImportError as e:
        st.error(f"Historical analyzer not available: {e}")
        return

    from datetime import date as _date, timedelta as _td

    # Date picker
    dc1, dc2 = st.columns([2, 1])
    with dc1:
        target_date = st.date_input(
            "Select Date",
            value=_date.today() - _td(days=1),
            max_value=_date.today(),
            min_value=_date(2020, 1, 1),
            key="hist_date",
        )
    with dc2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("Run Full Analysis", type="primary", key="run_hist_analysis")

    if not run_analysis:
        st.info("Select a date and click 'Run Full Analysis' to fetch real market data and compute all indicators.")
        return

    # Run analysis
    with st.spinner(f"Fetching real data for {target_date} ({display_name})..."):
        analysis = analyze_date(target_date, selected_index)

    # ── Data Sources & Warnings ──
    src_cols = st.columns(3)
    with src_cols[0]:
        st.metric("OHLC Data", "Available" if analysis.has_ohlc else "Not Available")
    with src_cols[1]:
        st.metric("India VIX", f"{analysis.vix_value:.1f}" if analysis.has_vix else "N/A")
    with src_cols[2]:
        st.metric("Option Chain", "Available" if analysis.has_options else "Not Available")

    if analysis.data_sources:
        st.caption(f"Data from: {', '.join(analysis.data_sources)}")
    if analysis.warnings:
        for w in analysis.warnings:
            st.warning(w)

    if not analysis.has_ohlc:
        st.error(f"No trading data for {target_date}. This may be a holiday or weekend.")
        return

    # ── Price Action ──
    st.markdown("---")
    st.markdown(f"### Price Action — {display_name} on {target_date}")
    tech = analysis.technical
    if tech:
        p1, p2, p3, p4, p5 = st.columns(5)
        change = tech.day_close - tech.prev_close
        change_pct = (change / tech.prev_close * 100) if tech.prev_close else 0
        p1.metric("Close", f"{tech.day_close:,.2f}", f"{change_pct:+.2f}%")
        p2.metric("Open", f"{tech.day_open:,.2f}")
        p3.metric("High", f"{tech.day_high:,.2f}")
        p4.metric("Low", f"{tech.day_low:,.2f}")
        p5.metric("Volume", f"{tech.volume:,}" if tech.volume else "N/A")

        # Gap
        if tech.gap_pct is not None:
            gap_str = f"{tech.gap_pct:+.2f}% ({tech.gap_direction})"
            st.caption(f"Gap: {gap_str}")

    # ── Technical Indicators ──
    st.markdown("---")
    st.markdown("### Technical Indicators (Real Data)")
    if tech:
        # EMAs
        st.markdown("**Moving Averages**")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("EMA 9", f"{tech.ema_9:,.2f}" if tech.ema_9 else "N/A",
                   delta="Above" if tech.ema_9 and tech.spot > tech.ema_9 else "Below")
        e2.metric("EMA 20", f"{tech.ema_20:,.2f}" if tech.ema_20 else "N/A",
                   delta="Above" if tech.ema_20 and tech.spot > tech.ema_20 else "Below")
        e3.metric("EMA 50", f"{tech.ema_50:,.2f}" if tech.ema_50 else "N/A",
                   delta="Above" if tech.ema_50 and tech.spot > tech.ema_50 else "Below")
        e4.metric("EMA 200", f"{tech.ema_200:,.2f}" if tech.ema_200 else "N/A",
                   delta="Above" if tech.ema_200 and tech.spot > tech.ema_200 else "Below")

        # Momentum & Trend
        st.markdown("**Momentum & Trend**")
        t1, t2, t3, t4 = st.columns(4)
        rsi_status = "Overbought" if tech.rsi_14 and tech.rsi_14 > 70 else ("Oversold" if tech.rsi_14 and tech.rsi_14 < 30 else "Neutral")
        t1.metric("RSI(14)", f"{tech.rsi_14:.1f}" if tech.rsi_14 else "N/A", delta=rsi_status)
        t2.metric("ADX(14)", f"{tech.adx:.1f}" if tech.adx else "N/A",
                   delta="Trending" if tech.adx and tech.adx > 25 else "Range-Bound")
        t3.metric("MACD", f"{tech.macd_line:.2f}" if tech.macd_line is not None else "N/A",
                   delta=f"Hist: {tech.macd_histogram:+.2f}" if tech.macd_histogram is not None else "")
        t4.metric("Supertrend", tech.supertrend_direction.title() if tech.supertrend_direction else "N/A")

        # Volatility
        st.markdown("**Volatility**")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("ATR(14)", f"{tech.atr_14:.2f}" if tech.atr_14 else "N/A")
        v2.metric("ATR %", f"{tech.atr_pct:.2f}%" if tech.atr_pct else "N/A")
        v3.metric("BB Width", f"{tech.bb_width_pct:.2f}%" if tech.bb_width_pct else "N/A")
        squeeze = "Yes" if tech.bb_width_pct and tech.bb_width_pct < 4 else "No"
        v4.metric("BB Squeeze", squeeze)

        # Bollinger Bands
        if tech.bb_upper:
            bb_pos = "Above Upper" if tech.spot > tech.bb_upper else (
                "Below Lower" if tech.spot < tech.bb_lower else "Within Bands")
            st.caption(
                f"Bollinger: {tech.bb_lower:,.0f} — {tech.bb_middle:,.0f} — {tech.bb_upper:,.0f} | "
                f"Price: {bb_pos}"
            )

        # VIX
        st.markdown("**Volatility Index**")
        vx1, vx2, vx3 = st.columns(3)
        vx1.metric("India VIX", f"{analysis.vix_value:.2f}" if analysis.has_vix else "N/A",
                     delta=f"{analysis.vix_change_pct:+.1f}% vs prev" if analysis.vix_change_pct else "")
        from core.market_regime import VIXRegime
        vix_regime = VIXRegime.classify(analysis.vix_value)
        vx2.metric("VIX Level", vix_regime.level.upper())
        vx3.metric("Sizing", f"{vix_regime.position_size_multiplier:.0%} of normal")
        st.caption(vix_regime.action)

        # Pivots
        if tech.pivot:
            st.markdown("**Pivot Levels (from previous day)**")
            pv1, pv2, pv3, pv4, pv5 = st.columns(5)
            pv1.metric("S2", f"{tech.s2:,.0f}")
            pv2.metric("S1", f"{tech.s1:,.0f}")
            pv3.metric("Pivot", f"{tech.pivot:,.0f}")
            pv4.metric("R1", f"{tech.r1:,.0f}")
            pv5.metric("R2", f"{tech.r2:,.0f}")

    # ── Options Analytics ──
    if analysis.has_options and analysis.options:
        st.markdown("---")
        st.markdown("### Options Analytics (Bhavcopy Data)")
        opts = analysis.options
        o1, o2, o3, o4 = st.columns(4)
        from core.market_regime import PCRRegime
        pcr_regime = PCRRegime.classify(opts.pcr)
        o1.metric("PCR", f"{opts.pcr:.2f}", delta=pcr_regime.bias.upper())
        o2.metric("Max Pain", f"{opts.max_pain:,.0f}" if opts.max_pain else "N/A")
        o3.metric("Highest Call OI", f"{opts.highest_call_oi_strike:,.0f}" if opts.highest_call_oi_strike else "N/A")
        o4.metric("Highest Put OI", f"{opts.highest_put_oi_strike:,.0f}" if opts.highest_put_oi_strike else "N/A")

        o5, o6, o7, o8 = st.columns(4)
        o5.metric("Total Call OI", f"{opts.total_call_oi:,}")
        o6.metric("Total Put OI", f"{opts.total_put_oi:,}")
        o7.metric("Call OI Chg", f"{opts.total_call_chg_oi:+,}")
        o8.metric("Put OI Chg", f"{opts.total_put_chg_oi:+,}")

        if opts.oi_buildup:
            st.caption(f"OI Buildup: {opts.oi_buildup.replace('_', ' ').title()}")
        st.caption(pcr_regime.interpretation)
        if opts.nearest_expiry:
            st.caption(f"Nearest expiry: {opts.nearest_expiry} | {opts.num_expiries} expiries available")

    # ── Day Classification ──
    if analysis.day_classification:
        st.markdown("---")
        st.markdown("### Day Classification")
        day_class = analysis.day_classification

        type_colors = {
            "trending": "#42a5f5", "range_bound": "#66bb6a",
            "volatile": "#ef5350", "expiry": "#ffa726",
        }
        type_labels = {
            "trending": "TRENDING", "range_bound": "RANGE-BOUND",
            "volatile": "VOLATILE", "expiry": "EXPIRY DAY",
        }
        color = type_colors.get(day_class.day_type, "#888")
        label = type_labels.get(day_class.day_type, day_class.day_type.upper())

        st.markdown(
            f'<div style="background:linear-gradient(135deg, {color}22, {color}44); '
            f'border:2px solid {color}; border-radius:12px; padding:16px; text-align:center;">'
            f'<h3 style="color:{color}; margin:0;">{label}</h3>'
            f'<p style="color:#ccc; margin:4px 0;">{day_class.summary}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Classification Factors", expanded=True):
            for f in day_class.factors:
                st.markdown(f"- {f}")

        # Strategy recommendations
        if analysis.recommendations:
            st.markdown("---")
            st.markdown("### Recommended Strategies for This Day")
            _render_recommendations(day_class)

    # ── OHLC Chart ──
    if analysis.has_ohlc and analysis.ohlc_df is not None:
        st.markdown("---")
        st.markdown("### Price History (last 30 days)")
        import plotly.graph_objects as go
        chart_data = analysis.ohlc_df.tail(30)
        fig = go.Figure(data=[go.Candlestick(
            x=[str(d) for d in chart_data["date"]],
            open=chart_data["open"], high=chart_data["high"],
            low=chart_data["low"], close=chart_data["close"],
            name=display_name,
        )])
        if tech and tech.ema_20:
            # Add EMA20 line on last 30 points
            ema20 = chart_data["close"].ewm(span=20, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=[str(d) for d in chart_data["date"]], y=ema20,
                mode="lines", name="EMA 20", line=dict(color="#ffa726", width=1),
            ))
        # Mark target date
        fig.add_vline(x=str(target_date), line_dash="dot", line_color="yellow", opacity=0.5)
        fig.update_layout(
            height=400, template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════


def render(selected_index: str, display_name: str, timeframe: str = "15m") -> None:
    """Render the Market Advisor tab."""
    st.header("Market Advisor")
    st.caption(
        "Day classification, strategy recommendations, historical analysis, "
        "and comprehensive strategy import. Uses real VIX, PCR, ADX, "
        "gap analysis, and CPR data from yfinance and bhavcopy."
    )

    # Tabs within the advisor
    advisor_tabs = st.tabs([
        "Live Regime",
        "Analyse Any Date",
        "Import Strategy JSON",
    ])

    with advisor_tabs[0]:
        day_class = _render_regime_dashboard(selected_index, display_name, timeframe)
        if day_class:
            st.divider()
            _render_recommendations(day_class)

    with advisor_tabs[1]:
        _render_historical_analysis(selected_index, display_name)

    with advisor_tabs[2]:
        _render_json_import()
