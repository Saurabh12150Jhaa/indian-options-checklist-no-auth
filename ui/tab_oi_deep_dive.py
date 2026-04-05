"""OI Deep Dive tab – multi-expiry OI comparison, timeline, and ITM/OTM analysis."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import INDEX_CONFIG


def render(selected_index: str, display_name: str) -> None:
    """Render the OI Deep Dive tab content."""

    st.header(f"OI Deep Dive – {display_name}")

    try:
        from oi_tracker import get_oi_timeline, get_aggregate_oi_timeline, get_tracked_dates
        from analysis import parse_nse_option_chain, get_expiry_dates as get_expiries

        # Get shared data from session state
        nse_data = st.session_state.get("_checklist_nse_data")
        ltp = st.session_state.get("_checklist_ltp", 0)
        options_result = st.session_state.get("_checklist_options_result", {})

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
