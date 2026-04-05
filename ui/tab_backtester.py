"""Strategy Backtester tab – strategy selection, custom builder, data management, and backtest execution."""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from config import INDEX_CONFIG


# ── Helper: render a single condition row ──────────────────────────────────

def _render_condition_row(i: int, registry: dict, prefix: str = "cond") -> dict:
    """Render Streamlit widgets for one condition and return its config dict."""
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        cond_type = st.selectbox(
            "Indicator",
            list(registry.keys()),
            key=f"{prefix}_type_{i}",
            format_func=lambda x: registry[x]["label"],
        )
    cond_info = registry[cond_type]
    desc = cond_info.get("description", "")
    if desc:
        st.caption(desc)
    params = {}
    with cc2:
        # Render params in a compact row
        param_cols = st.columns(len(cond_info["params"])) if len(cond_info["params"]) > 1 else [cc2]
        for j, p in enumerate(cond_info["params"]):
            col = param_cols[j] if len(cond_info["params"]) > 1 else cc2
            pk = f"{prefix}_{i}_{p['name']}"
            with col:
                if p["type"] == "int":
                    params[p["name"]] = st.number_input(
                        p["label"], p.get("min", 1), p.get("max", 200), p["default"], key=pk)
                elif p["type"] == "float":
                    params[p["name"]] = st.number_input(
                        p["label"], float(p.get("min", 0.0)), float(p.get("max", 100.0)),
                        float(p["default"]), step=0.1, key=pk)
                elif p["type"] == "select":
                    params[p["name"]] = st.selectbox(
                        p["label"], p["options"],
                        index=p["options"].index(p["default"]), key=pk)
                elif p["type"] == "multiselect":
                    params[p["name"]] = st.multiselect(
                        p["label"], p["options"], default=p["default"], key=pk)
    return {"type": cond_type, "params": params}


# ── Helper: render a single leg row ───────────────────────────────────────

def _render_leg_row(i: int, templates: dict, prefix: str = "leg") -> dict:
    """Render Streamlit widgets for one option leg and return its config dict."""
    lc1, lc2, lc3, lc4 = st.columns([2, 1, 1, 1])
    with lc1:
        strike_mode = st.selectbox(
            f"Leg {i+1} Strike",
            ["template", "custom_pct", "custom_points"],
            key=f"{prefix}_mode_{i}",
            format_func=lambda x: {"template": "Preset Strike", "custom_pct": "Custom % Offset",
                                    "custom_points": "Custom Points Offset"}[x],
        )
    leg = {}
    if strike_mode == "template":
        with lc1:
            template = st.selectbox(
                "Strike Position",
                list(templates.keys()),
                key=f"{prefix}_tmpl_{i}",
                format_func=lambda x: templates[x]["label"],
            )
        leg["template"] = template
    elif strike_mode == "custom_pct":
        with lc1:
            opt_type = st.selectbox("Type", ["CE", "PE"], key=f"{prefix}_ctype_{i}")
            offset_pct = st.number_input(
                "Offset %", -10.0, 10.0, 0.0, step=0.5,
                key=f"{prefix}_cpct_{i}",
                help="0 = ATM, +2 = 2% OTM call / ITM put, -2 = 2% ITM call / OTM put",
            )
        leg["option_type"] = opt_type
        leg["offset_pct"] = offset_pct
    else:  # custom_points
        with lc1:
            opt_type = st.selectbox("Type", ["CE", "PE"], key=f"{prefix}_ptype_{i}")
            offset_pts = st.number_input(
                "Offset Points", -2000, 2000, 0, step=50,
                key=f"{prefix}_ppts_{i}",
                help="0 = ATM, +100 = 100 points above spot, -200 = 200 points below",
            )
        leg["option_type"] = opt_type
        leg["offset_points"] = offset_pts

    with lc2:
        leg["action"] = st.selectbox("Action", ["BUY", "SELL"], key=f"{prefix}_action_{i}")
    with lc3:
        leg["qty"] = st.number_input("Qty", 1, 10, 1, key=f"{prefix}_qty_{i}")
    with lc4:
        # Show leg summary
        if "template" in leg:
            tmpl = templates.get(leg["template"], {})
            st.markdown(f"<br><small>{tmpl.get('label', '')}</small>", unsafe_allow_html=True)
        elif "offset_points" in leg:
            st.markdown(
                f"<br><small>{leg['option_type']} ATM{leg['offset_points']:+d}pts</small>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<br><small>{leg.get('option_type', 'CE')} ATM{leg.get('offset_pct', 0):+.1f}%</small>",
                unsafe_allow_html=True)
    return leg


# ── Helper: payoff preview ────────────────────────────────────────────────

def _render_payoff_preview(legs: list, templates: dict, spot_estimate: float = 24000):
    """Render an approximate payoff diagram from the leg config."""
    if not legs or spot_estimate <= 0:
        return
    # Build simplified payoff
    import numpy as np
    strikes = []
    for leg in legs:
        tmpl_name = leg.get("template")
        tmpl = templates.get(tmpl_name, {})
        offset = tmpl.get("offset_pct", 0)
        offset = leg.get("offset_pct", offset)
        if "offset_points" in leg and spot_estimate > 0:
            offset = (leg["offset_points"] / spot_estimate) * 100
        otype = tmpl.get("option_type", leg.get("option_type", "CE"))
        strike = spot_estimate * (1 + offset / 100)
        strikes.append({
            "strike": round(strike, 0),
            "type": otype,
            "action": leg.get("action", "BUY"),
            "qty": leg.get("qty", 1),
        })

    if not strikes:
        return

    lo = min(s["strike"] for s in strikes) * 0.95
    hi = max(s["strike"] for s in strikes) * 1.05
    x = np.linspace(lo, hi, 200)
    payoff = np.zeros_like(x)
    total_premium = 0
    est_premium = spot_estimate * 0.015  # rough 1.5% premium estimate

    for s in strikes:
        k = s["strike"]
        q = s["qty"]
        sign = 1 if s["action"] == "BUY" else -1
        prem = est_premium * q
        if s["type"] == "CE":
            intrinsic = np.maximum(x - k, 0) * q * sign
        else:
            intrinsic = np.maximum(k - x, 0) * q * sign
        payoff += intrinsic
        total_premium += prem * sign

    payoff -= total_premium

    fig = go.Figure()
    colors = ["#66bb6a" if v >= 0 else "#ef5350" for v in payoff]
    fig.add_trace(go.Scatter(
        x=x, y=payoff, mode="lines", name="P&L",
        line=dict(color="#42a5f5", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_vline(x=spot_estimate, line_dash="dot", line_color="yellow", opacity=0.4,
                  annotation_text="Spot", annotation_font_color="yellow")
    for s in strikes:
        fig.add_vline(x=s["strike"], line_dash="dot", line_color="#888", opacity=0.3)
    fig.update_layout(
        title="Approximate Payoff at Expiry", height=250,
        template="plotly_dark", margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Underlying Price", yaxis_title="P&L",
    )
    st.plotly_chart(fig, width="stretch")


def _render_backtest_results(report, config) -> None:
    """Render backtest results (metrics, equity curve, trade log).

    Extracted as a helper so it can be called both inside the Run-button
    block *and* on subsequent reruns from session-state, fixing the
    'results vanish on tab switch' bug.
    """
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


def render() -> None:
    """Render the Strategy Backtester tab content."""

    st.header("Strategy Backtester")

    try:
        from backtester.strategies import PREBUILT_STRATEGIES, get_strategy_fn
        from backtester.smart_money import SMC_STRATEGIES
        from backtester.price_action import PA_STRATEGIES
        from backtester.custom_strategy import (
            CustomStrategy, CUSTOM_PRESETS, CONDITION_REGISTRY, LEG_TEMPLATES,
            describe_strategy,
        )

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
                "description": describe_strategy(preset),
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

        # ══════════════════════════════════════════════════════════════
        # CUSTOM STRATEGY BUILDER — 3 modes
        # ══════════════════════════════════════════════════════════════

        with st.expander("Build Custom Strategy", expanded=False):
            builder_mode = st.radio(
                "Builder Mode",
                ["Quick Presets", "Guided Builder", "Advanced Builder"],
                horizontal=True,
                key="builder_mode",
                help="Quick Presets: one-click strategies for beginners. "
                     "Guided: step-by-step with explanations. "
                     "Advanced: full control with condition groups, custom strikes, JSON export.",
            )

            # ── MODE 1: Quick Presets ─────────────────────────────────
            if builder_mode == "Quick Presets":
                st.markdown("Pick a ready-made strategy. Each is battle-tested and works out of the box.")

                # Group presets by market view
                preset_groups = {
                    "Bullish": [],
                    "Bearish": [],
                    "Neutral / Range-Bound": [],
                    "Volatility": [],
                    "Expiry Day": [],
                }
                for name, preset in CUSTOM_PRESETS.items():
                    n = name.lower()
                    if any(w in n for w in ("bull", "long_buildup", "recovery", "momentum", "green")):
                        preset_groups["Bullish"].append((name, preset))
                    elif any(w in n for w in ("bear", "short_buildup")):
                        preset_groups["Bearish"].append((name, preset))
                    elif any(w in n for w in ("straddle sell", "strangle sell", "iron", "range", "squeeze", "low adx")):
                        preset_groups["Neutral / Range-Bound"].append((name, preset))
                    elif any(w in n for w in ("expiry", "thursday")):
                        preset_groups["Expiry Day"].append((name, preset))
                    else:
                        preset_groups["Volatility"].append((name, preset))

                for group_name, presets in preset_groups.items():
                    if not presets:
                        continue
                    st.markdown(f"**{group_name}**")
                    for pname, preset in presets:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            desc = describe_strategy(preset)
                            st.markdown(
                                f'<div style="background:#1e1e2f; padding:10px 14px; border-radius:8px; '
                                f'margin:4px 0; border-left:3px solid #42a5f5;">'
                                f'<strong>{pname}</strong><br/>'
                                f'<span style="color:#aaa; font-size:0.85rem;">{desc}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with col2:
                            if st.button("Use", key=f"qp_{pname}"):
                                st.session_state["custom_strategy"] = preset
                                st.success(f"Loaded: {pname}")

            # ── MODE 2: Guided Builder ────────────────────────────────
            elif builder_mode == "Guided Builder":
                st.markdown(
                    "Build your strategy step by step. Each section has explanations "
                    "to help you understand what you're configuring."
                )

                custom_name = st.text_input("Strategy Name", "My Custom Strategy", key="guided_name")

                # Step 1: Market View
                st.markdown("---")
                st.markdown("#### Step 1: When should we enter?")
                st.caption(
                    "Pick one or more conditions. The strategy will only enter a trade "
                    "when ALL conditions are true at the same time."
                )

                num_conditions = st.slider(
                    "How many conditions?", 1, 5, 2, key="guided_num_conds",
                    help="Start with 1-2 conditions. More conditions = fewer trades but higher conviction."
                )

                conditions = []
                for i in range(num_conditions):
                    st.markdown(f"**Condition {i+1}**")
                    cond = _render_condition_row(i, CONDITION_REGISTRY, prefix="guided_cond")
                    conditions.append(cond)

                # Step 2: What to trade
                st.markdown("---")
                st.markdown("#### Step 2: What should we trade?")
                st.caption(
                    "Define the option legs. For a simple directional trade, use 1 leg. "
                    "For spreads (limited risk), use 2 legs. For iron condors / butterflies, use 4."
                )

                # Quick leg presets for beginners
                leg_preset = st.selectbox(
                    "Quick setup",
                    [
                        "Custom (build your own)",
                        "Buy Call (bullish)",
                        "Buy Put (bearish)",
                        "Bull Call Spread (limited risk bullish)",
                        "Bear Put Spread (limited risk bearish)",
                        "Short Straddle (neutral, sell volatility)",
                        "Short Strangle (neutral, wider range)",
                        "Iron Condor (neutral, defined risk)",
                    ],
                    key="guided_leg_preset",
                )

                # Map presets to leg configs
                _QUICK_LEGS = {
                    "Buy Call (bullish)": [
                        {"template": "atm_call", "action": "BUY", "qty": 1},
                    ],
                    "Buy Put (bearish)": [
                        {"template": "atm_put", "action": "BUY", "qty": 1},
                    ],
                    "Bull Call Spread (limited risk bullish)": [
                        {"template": "atm_call", "action": "BUY", "qty": 1},
                        {"template": "otm_call_1", "action": "SELL", "qty": 1},
                    ],
                    "Bear Put Spread (limited risk bearish)": [
                        {"template": "atm_put", "action": "BUY", "qty": 1},
                        {"template": "otm_put_1", "action": "SELL", "qty": 1},
                    ],
                    "Short Straddle (neutral, sell volatility)": [
                        {"template": "atm_call", "action": "SELL", "qty": 1},
                        {"template": "atm_put", "action": "SELL", "qty": 1},
                    ],
                    "Short Strangle (neutral, wider range)": [
                        {"template": "otm_call_1", "action": "SELL", "qty": 1},
                        {"template": "otm_put_1", "action": "SELL", "qty": 1},
                    ],
                    "Iron Condor (neutral, defined risk)": [
                        {"template": "otm_call_1", "action": "SELL", "qty": 1},
                        {"template": "otm_call_2", "action": "BUY", "qty": 1},
                        {"template": "otm_put_1", "action": "SELL", "qty": 1},
                        {"template": "otm_put_2", "action": "BUY", "qty": 1},
                    ],
                }

                if leg_preset != "Custom (build your own)" and leg_preset in _QUICK_LEGS:
                    legs = _QUICK_LEGS[leg_preset]
                    st.caption(f"Using preset: {leg_preset}")
                    for li, leg in enumerate(legs):
                        tmpl = LEG_TEMPLATES.get(leg["template"], {})
                        action_color = "#66bb6a" if leg["action"] == "BUY" else "#ef5350"
                        st.markdown(
                            f'<span style="color:{action_color}; font-weight:700;">'
                            f'{leg["action"]}</span> {leg["qty"]}x {tmpl.get("label", leg["template"])}',
                            unsafe_allow_html=True,
                        )
                else:
                    num_legs = st.slider("Number of legs", 1, 6, 2, key="guided_num_legs")
                    legs = []
                    for li in range(num_legs):
                        leg = _render_leg_row(li, LEG_TEMPLATES, prefix="guided_leg")
                        legs.append(leg)

                # Payoff preview
                st.markdown("---")
                st.markdown("#### Payoff Preview")
                _render_payoff_preview(legs, LEG_TEMPLATES)

                # Build & use
                if st.button("Use This Strategy", type="primary", key="guided_use"):
                    config = {
                        "name": custom_name,
                        "conditions": conditions,
                        "condition_logic": "AND",
                        "legs": legs,
                    }
                    st.session_state["custom_strategy"] = config
                    st.success(f"Strategy '{custom_name}' is ready! Configure backtest settings below and hit Run.")

            # ── MODE 3: Advanced Builder ──────────────────────────────
            else:
                st.markdown("Full control: condition groups with nested logic, custom strike offsets, JSON export/import.")

                custom_name = st.text_input("Strategy Name", "My Advanced Strategy", key="adv_name")

                # ── Condition Groups ──
                st.markdown("---")
                st.markdown("#### Entry Conditions (with groups)")
                st.caption(
                    "Group conditions with AND/OR logic. Groups combine with their own logic. "
                    "Example: (RSI < 30 AND Price > EMA200) **OR** (Bullish Engulfing AND PCR > 1.2)"
                )

                num_groups = st.number_input("Condition groups", 1, 4, 1, key="adv_num_groups")
                group_logic = st.radio(
                    "How should groups combine?", ["AND", "OR"],
                    horizontal=True, key="adv_group_logic",
                    help="AND = all groups must be true. OR = any one group being true is enough.",
                )

                condition_groups = []
                for g in range(int(num_groups)):
                    st.markdown(f"**Group {g+1}**")
                    g_logic = st.radio(
                        f"Group {g+1} internal logic", ["AND", "OR"],
                        horizontal=True, key=f"adv_grp_logic_{g}",
                    )
                    g_num = st.number_input(
                        f"Conditions in group {g+1}", 1, 6, 2, key=f"adv_grp_num_{g}")
                    g_conds = []
                    for ci in range(int(g_num)):
                        cond = _render_condition_row(ci, CONDITION_REGISTRY, prefix=f"adv_g{g}_c")
                        g_conds.append(cond)
                    condition_groups.append({"conditions": g_conds, "logic": g_logic})

                # ── Legs ──
                st.markdown("---")
                st.markdown("#### Option Legs")
                st.caption(
                    "Use preset strikes, custom % offset, or exact point offset from ATM. "
                    "Up to 6 legs for complex strategies."
                )

                num_legs = st.number_input("Number of legs", 1, 6, 2, key="adv_num_legs")
                legs = []
                for li in range(int(num_legs)):
                    leg = _render_leg_row(li, LEG_TEMPLATES, prefix="adv_leg")
                    legs.append(leg)

                # Payoff preview
                st.markdown("---")
                st.markdown("#### Payoff Preview")
                _render_payoff_preview(legs, LEG_TEMPLATES)

                # Strategy summary
                config = {
                    "name": custom_name,
                    "condition_groups": condition_groups,
                    "group_logic": group_logic,
                    "legs": legs,
                }
                st.markdown("---")
                st.markdown("#### Strategy Summary")
                try:
                    st.markdown(describe_strategy(config))
                except Exception:
                    pass

                # Action buttons
                btn1, btn2, btn3 = st.columns(3)
                with btn1:
                    if st.button("Use This Strategy", type="primary", key="adv_use"):
                        st.session_state["custom_strategy"] = config
                        st.success(f"Strategy '{custom_name}' is ready!")

                with btn2:
                    # Export as JSON
                    json_str = json.dumps(config, indent=2, default=str)
                    st.download_button(
                        "Export JSON",
                        data=json_str,
                        file_name=f"{custom_name.replace(' ', '_').lower()}.json",
                        mime="application/json",
                        key="adv_export",
                    )

                with btn3:
                    # Import JSON
                    uploaded = st.file_uploader("Import JSON", type=["json"], key="adv_import")
                    if uploaded:
                        try:
                            imported = json.loads(uploaded.read())
                            st.session_state["custom_strategy"] = imported
                            st.success(f"Imported: {imported.get('name', 'Unknown')}")
                        except Exception as e:
                            st.error(f"Invalid JSON: {e}")

        # Show active custom strategy if set
        active_custom = st.session_state.get("custom_strategy")
        if active_custom:
            try:
                summary = describe_strategy(active_custom)
            except Exception:
                summary = active_custom.get("name", "Custom Strategy")
            st.success(f"Active custom strategy: {summary}")
            if st.button("Clear custom strategy", key="clear_custom"):
                del st.session_state["custom_strategy"]
                st.rerun()

        st.divider()

        # Backtest configuration
        st.subheader("Backtest Configuration")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bt_symbol = st.selectbox("Symbol", list(INDEX_CONFIG.keys()) + ["RELIANCE", "INFY", "HDFCBANK"], key="bt_symbol")
        with bc2:
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
                                # Ensure underlying_close exists (old-format bhavcopies lack it)
                                if "underlying_close" not in options_df.columns or options_df["underlying_close"].isna().all():
                                    try:
                                        from backtester.data_adapter import merge_underlying_prices
                                        from backtester.utils import _fetch_real_ohlc
                                        ohlc = _fetch_real_ohlc(bt_symbol, bt_start, bt_end)
                                        if ohlc is not None and not ohlc.empty:
                                            ohlc = ohlc.rename(columns={"close": "underlying_close"})
                                            ohlc["date"] = pd.to_datetime(ohlc["date"])
                                            options_df["date"] = pd.to_datetime(options_df["date"])
                                            options_df = options_df.merge(
                                                ohlc[["date", "underlying_close"]].drop_duplicates("date"),
                                                on="date", how="left", suffixes=("_old", ""),
                                            )
                                            if "underlying_close_old" in options_df.columns:
                                                options_df.drop(columns=["underlying_close_old"], inplace=True)
                                    except Exception:
                                        pass  # proceed without underlying — engine handles None

                                data_found = True
                                # Mark files as accessed so they aren't cleaned up
                                mark_files_accessed(csv_files)

                                # Get lot size — check INDEX_CONFIG first, then FNO_STOCKS
                                lot = INDEX_CONFIG.get(bt_symbol, {}).get("lot_size", 0)
                                if lot == 0:
                                    from config import FNO_STOCKS
                                    lot = FNO_STOCKS.get(bt_symbol, {}).get("lot_size", 25)

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

                                    # Store results in session state for persistence across tab switches
                                    st.session_state["backtest_results"] = {
                                        "report": report,
                                        "config": config,
                                        "strategy_name": use_custom["name"] if use_custom else sel_strategy,
                                        "symbol": bt_symbol,
                                        "start": str(bt_start),
                                        "end": str(bt_end),
                                    }

                                    # Display results
                                    st.subheader("Backtest Results")
                                    _render_backtest_results(report, config)
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

        # Persist results across reruns (when button is not clicked)
        elif "backtest_results" in st.session_state:
            results = st.session_state["backtest_results"]
            st.subheader(f"Backtest Results: {results['strategy_name']} on {results['symbol']}")
            st.caption(f"Period: {results['start']} to {results['end']}")
            _render_backtest_results(results["report"], results["config"])
            if st.button("Clear Results", key="clear_results"):
                del st.session_state["backtest_results"]
                st.rerun()

    except ImportError as e:
        st.info(f"Backtester modules loading: {e}")
