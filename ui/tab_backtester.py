"""Strategy Backtester tab – strategy selection, custom builder, data management, and backtest execution."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from config import INDEX_CONFIG


def render() -> None:
    """Render the Strategy Backtester tab content."""

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
