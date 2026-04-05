"""
Domain constants and type aliases.

Centralises signal constants and common type shapes used across analysis,
services, and UI layers.  Keeping these in one place satisfies the
Single-Responsibility Principle and avoids duplicated magic strings.
"""

from __future__ import annotations

# ── Signal direction constants ────────────────────────────────────────────
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"
