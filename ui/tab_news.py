"""Market News tab – news feed with sentiment filtering."""

import streamlit as st
import plotly.graph_objects as go

from ui.cache import _news
from ui.components import freshness


def render(selected_index: str, display_name: str) -> None:
    """Render the Market News tab content."""

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
