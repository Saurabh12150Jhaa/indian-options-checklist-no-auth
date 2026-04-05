"""Stock Options Analysis tab."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.components import freshness


def render() -> None:
    """Render the Stock Options Analysis tab content."""

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
