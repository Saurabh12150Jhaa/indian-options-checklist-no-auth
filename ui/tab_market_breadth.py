"""Market Breadth tab – advances/declines, sector performance, top movers, turnover."""

import streamlit as st
import plotly.graph_objects as go

from ui.components import freshness


def render() -> None:
    """Render the Market Breadth tab content."""

    st.header("Market Breadth & Internals")

    try:
        from market_data import (
            fetch_top_gainers, fetch_top_losers,
            fetch_advances_declines, fetch_pre_open,
            fetch_sector_performance, fetch_market_turnover,
        )

        # ── Row 1: Advances / Declines + Breadth Gauge ──────────────
        st.subheader("Market Breadth")
        with st.spinner("Fetching market breadth..."):
            ad_data, ad_ts = fetch_advances_declines()

        if ad_data:
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            bc1.metric("Advances", f"{ad_data['advances']:,}")
            bc2.metric("Declines", f"{ad_data['declines']:,}")
            bc3.metric("Unchanged", f"{ad_data['unchanged']:,}")
            bc4.metric("A/D Ratio", ad_data["ad_ratio"])

            # Breadth sentiment
            ratio = ad_data["ad_ratio"]
            if ratio >= 2.0:
                sentiment = "Strong Bullish"
                color = "#4caf50"
            elif ratio >= 1.3:
                sentiment = "Bullish"
                color = "#66bb6a"
            elif ratio >= 0.75:
                sentiment = "Neutral"
                color = "#fdd835"
            elif ratio >= 0.5:
                sentiment = "Bearish"
                color = "#ff7043"
            else:
                sentiment = "Strong Bearish"
                color = "#ef5350"
            bc5.markdown(
                f"<div style='text-align:center;padding:8px;'>"
                f"<span style='font-size:0.8rem;color:#888;'>Sentiment</span><br>"
                f"<span style='font-size:1.3rem;font-weight:bold;color:{color};'>"
                f"{sentiment}</span></div>",
                unsafe_allow_html=True,
            )

            # Donut chart + bar chart side by side
            ch1, ch2 = st.columns(2)
            with ch1:
                ad_fig = go.Figure(data=[
                    go.Pie(
                        labels=["Advances", "Declines", "Unchanged"],
                        values=[ad_data["advances"], ad_data["declines"], ad_data["unchanged"]],
                        marker=dict(colors=["#66bb6a", "#ef5350", "#fdd835"]),
                        hole=0.5,
                        textinfo="label+percent",
                        textfont_size=12,
                    )
                ])
                ad_fig.update_layout(
                    height=280, template="plotly_dark",
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(ad_fig, width="stretch")

            with ch2:
                # Horizontal bar: breadth strength indicator
                total = ad_data["total"] or 1
                adv_pct = round(ad_data["advances"] / total * 100, 1)
                dec_pct = round(ad_data["declines"] / total * 100, 1)
                unc_pct = round(ad_data["unchanged"] / total * 100, 1)

                bar_fig = go.Figure()
                bar_fig.add_trace(go.Bar(
                    y=["Market"], x=[adv_pct], name="Advances",
                    orientation="h", marker_color="#66bb6a",
                    text=f"{adv_pct}%", textposition="inside",
                ))
                bar_fig.add_trace(go.Bar(
                    y=["Market"], x=[unc_pct], name="Unchanged",
                    orientation="h", marker_color="#fdd835",
                    text=f"{unc_pct}%", textposition="inside",
                ))
                bar_fig.add_trace(go.Bar(
                    y=["Market"], x=[dec_pct], name="Declines",
                    orientation="h", marker_color="#ef5350",
                    text=f"{dec_pct}%", textposition="inside",
                ))
                bar_fig.update_layout(
                    barmode="stack", height=120,
                    template="plotly_dark",
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True, legend=dict(orientation="h", y=-0.3),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(bar_fig, width="stretch")

                # Market value metrics if available
                tv = ad_data.get("totalTradedValue")
                mc = ad_data.get("totalMarketCap")
                if tv or mc:
                    v1, v2 = st.columns(2)
                    if tv:
                        try:
                            tv_val = float(tv) / 100  # Convert to Cr from Lakhs
                            v1.metric("Turnover", f"{tv_val:,.0f} Cr")
                        except (ValueError, TypeError):
                            pass
                    if mc:
                        try:
                            v2.metric("Market Cap", f"{mc}")
                        except (ValueError, TypeError):
                            pass

            st.markdown(freshness(ad_ts), unsafe_allow_html=True)
        else:
            st.info("Market breadth data unavailable.")

        # ── Row 2: Sector Performance ────────────────────────────────
        st.subheader("Sector Performance")
        with st.spinner("Fetching sector data..."):
            sector_df, sector_ts = fetch_sector_performance()

        if sector_df is not None and not sector_df.empty:
            # Color-coded bar chart
            colors = [
                "#66bb6a" if v >= 0 else "#ef5350"
                for v in sector_df["Change"]
            ]
            sector_fig = go.Figure(data=[
                go.Bar(
                    x=sector_df["Change"],
                    y=sector_df["Sector"].str.replace("NIFTY ", ""),
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f}%" for v in sector_df["Change"]],
                    textposition="outside",
                    textfont_size=11,
                )
            ])
            sector_fig.update_layout(
                height=max(400, len(sector_df) * 28),
                template="plotly_dark",
                margin=dict(l=120, r=50, t=10, b=10),
                xaxis_title="% Change",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(sector_fig, width="stretch")
            st.markdown(freshness(sector_ts), unsafe_allow_html=True)
        else:
            st.info("Sector performance data unavailable.")

        # ── Row 3: Top Gainers / Losers ──────────────────────────────
        g_col, l_col = st.columns(2)

        with g_col:
            st.subheader("Top Gainers (NIFTY 50)")
            with st.spinner("Loading..."):
                gainers_df, g_ts = fetch_top_gainers()
            if gainers_df is not None and not gainers_df.empty:
                st.dataframe(
                    gainers_df.style.format({
                        "LTP": "{:,.2f}",
                        "Change": "{:+.2f}",
                        "Change %": "{:+.2f}",
                        "Volume": "{:,.0f}",
                    }).applymap(
                        lambda v: "color: #66bb6a" if isinstance(v, (int, float)) and v > 0 else "",
                        subset=["Change %"],
                    ),
                    width="stretch", hide_index=True,
                )
            else:
                st.info("No gainers in NIFTY 50 today.")

        with l_col:
            st.subheader("Top Losers (NIFTY 50)")
            with st.spinner("Loading..."):
                losers_df, l_ts = fetch_top_losers()
            if losers_df is not None and not losers_df.empty:
                st.dataframe(
                    losers_df.style.format({
                        "LTP": "{:,.2f}",
                        "Change": "{:+.2f}",
                        "Change %": "{:+.2f}",
                        "Volume": "{:,.0f}",
                    }).applymap(
                        lambda v: "color: #ef5350" if isinstance(v, (int, float)) and v < 0 else "",
                        subset=["Change %"],
                    ),
                    width="stretch", hide_index=True,
                )
            else:
                st.info("No losers in NIFTY 50 today.")

        # ── Row 4: Pre-open data ─────────────────────────────────────
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
