"""Shared UI components, badges, and styling for the Streamlit dashboard."""

from datetime import datetime

import streamlit as st


# ── Custom CSS injection ───────────────────────────────────────────────────


def inject_custom_css() -> None:
    """Inject the global custom CSS styles into the Streamlit page."""
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


# ── Helper functions ───────────────────────────────────────────────────────


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
    return f'<span class="freshness">Updated: {dt.strftime("%d %b %Y, %H:%M:%S IST")}</span>'


def phase_badge(phase: str) -> str:
    labels = {"pre_market": "PRE-MARKET", "market_open": "MARKET OPEN", "post_market": "POST-MARKET"}
    return f'<span class="phase-badge phase-{phase}">{labels.get(phase, phase.upper())}</span>'


def colour_change(val):
    """Style helper for colouring positive/negative numbers."""
    if isinstance(val, (int, float)):
        c = "#4caf50" if val >= 0 else "#ef5350"
        return f"color: {c}; font-weight: 700"
    return ""


def _build_checklist_html(items: list[dict]) -> str:
    """Build a clean HTML table for the checklist."""
    signal_styles = {
        "BULLISH": "background:#1b5e20; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;",
        "BEARISH": "background:#b71c1c; color:#fff; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;",
        "NEUTRAL": "background:#f57f17; color:#000; padding:3px 10px; border-radius:4px; font-weight:700; font-size:0.8rem;",
    }
    weight_bar_max = max(i["weight"] for i in items) if items else 1

    rows_html = ""
    for i, item in enumerate(items):
        sig = item["signal"]
        sig_html = f'<span style="{signal_styles.get(sig, signal_styles["NEUTRAL"])}">{sig}</span>'

        # Weight as a visual bar
        bar_pct = (item["weight"] / weight_bar_max) * 100
        bar_colour = "#42a5f5"
        weight_html = (
            f'<div style="display:flex; align-items:center; gap:6px;">'
            f'<div style="background:{bar_colour}; height:8px; width:{bar_pct:.0f}%; border-radius:4px; min-width:10px;"></div>'
            f'<span style="color:#ccc; font-size:0.8rem;">{item["weight"]:.1f}</span>'
            f'</div>'
        )

        # Row background alternation
        bg = "#1a1a2e" if i % 2 == 0 else "#16213e"

        # Value styling — colour hint based on signal
        val_color = {"BULLISH": "#66bb6a", "BEARISH": "#ef5350"}.get(sig, "#e0e0e0")

        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:10px 14px; font-weight:600; color:#e0e0e0; border-bottom:1px solid #2a2a4a;">'
            f'<span style="color:#aaa; font-size:0.75rem; margin-right:6px;">#{i+1}</span>{item["indicator"]}</td>'
            f'<td style="padding:10px 14px; color:{val_color}; font-weight:500; border-bottom:1px solid #2a2a4a;">{item["value"]}</td>'
            f'<td style="padding:10px 14px; border-bottom:1px solid #2a2a4a;">{sig_html}</td>'
            f'<td style="padding:10px 14px; border-bottom:1px solid #2a2a4a;">{weight_html}</td>'
            f'</tr>'
        )

    return (
        f'<table style="width:100%; border-collapse:collapse; border-radius:10px; overflow:hidden; '
        f'border:1px solid #2a2a4a; margin-top:8px;">'
        f'<thead>'
        f'<tr style="background:#0d1b2a;">'
        f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Indicator</th>'
        f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Value</th>'
        f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Signal</th>'
        f'<th style="padding:10px 14px; text-align:left; color:#90caf9; font-size:0.85rem; border-bottom:2px solid #42a5f5;">Weight</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )
