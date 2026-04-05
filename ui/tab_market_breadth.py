"""Market Breadth tab – advances/declines, top gainers/losers, pre-open data."""

import streamlit as st
import plotly.graph_objects as go

from ui.components import freshness


def render() -> None:
    """Render the Market Breadth tab content."""

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
