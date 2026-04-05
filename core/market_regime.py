"""
Market Regime Detector & Strategy Advisor.

Classifies the trading day into regimes (trending, range-bound, volatile, expiry)
based on VIX, ADX, ORB range, gap analysis, CPR, and OI data.
Recommends strategies appropriate for the detected regime.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from config import IST, INDEX_CONFIG

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class VIXRegime:
    """India VIX classification."""
    value: float
    level: str  # low / normal / high / extreme
    action: str
    position_size_multiplier: float

    @staticmethod
    def classify(vix: float) -> "VIXRegime":
        if vix <= 0:
            return VIXRegime(vix, "unknown", "No VIX data", 1.0)
        if vix < 13:
            return VIXRegime(vix, "low", "Sell premium strategies preferred. Low volatility = low premiums but high probability.", 1.0)
        if vix < 18:
            return VIXRegime(vix, "normal", "Both buying and selling strategies viable. Normal market conditions.", 1.0)
        if vix < 25:
            return VIXRegime(vix, "high", "Directional buying or wide strangles. Premiums are rich — selling is lucrative but risky.", 0.75)
        return VIXRegime(vix, "extreme", "Reduce position size by 50%. Hedged strategies only. Expect wild swings.", 0.5)


@dataclass
class PCRRegime:
    """Put-Call Ratio classification."""
    value: float
    interpretation: str
    bias: str  # bullish / bearish / neutral

    @staticmethod
    def classify(pcr: float) -> "PCRRegime":
        if pcr <= 0:
            return PCRRegime(pcr, "No data", "neutral")
        if pcr < 0.7:
            return PCRRegime(pcr, "Bearish — more calls written, market may fall", "bearish")
        if pcr < 1.0:
            return PCRRegime(pcr, "Neutral to mildly bullish", "neutral")
        if pcr < 1.5:
            return PCRRegime(pcr, "Bullish — more puts written, market well supported", "bullish")
        return PCRRegime(pcr, "Extremely bullish or potential reversal warning — stretched sentiment", "bullish")


@dataclass
class GapAnalysis:
    """Opening gap classification."""
    gap_pct: float
    direction: str  # gap_up / gap_down / flat
    magnitude: str  # small / medium / large

    @staticmethod
    def classify(prev_close: float, current_open: float) -> "GapAnalysis":
        if prev_close <= 0:
            return GapAnalysis(0.0, "flat", "none")
        gap_pct = ((current_open - prev_close) / prev_close) * 100
        if abs(gap_pct) < 0.2:
            direction = "flat"
            magnitude = "none"
        elif gap_pct > 0:
            direction = "gap_up"
            magnitude = "small" if gap_pct < 0.5 else ("medium" if gap_pct < 1.0 else "large")
        else:
            direction = "gap_down"
            magnitude = "small" if abs(gap_pct) < 0.5 else ("medium" if abs(gap_pct) < 1.0 else "large")
        return GapAnalysis(round(gap_pct, 2), direction, magnitude)


@dataclass
class ORBAnalysis:
    """Opening Range Breakout analysis (first N-minute candle)."""
    orb_high: float
    orb_low: float
    orb_range: float
    orb_range_pct: float
    assessment: str  # narrow / normal / wide
    tradeable: bool

    @staticmethod
    def from_ohlc(ohlc_first_candle: dict, spot: float, instrument: str = "NIFTY") -> "ORBAnalysis":
        """
        ohlc_first_candle: dict with keys high, low, open, close
        spot: current underlying price for percentage calc
        instrument: NIFTY or BANKNIFTY for range thresholds
        """
        h = ohlc_first_candle.get("high", 0)
        l = ohlc_first_candle.get("low", 0)
        orb_range = h - l
        orb_range_pct = (orb_range / spot * 100) if spot > 0 else 0

        # Instrument-specific range thresholds
        if instrument.upper() in ("BANKNIFTY", "BANK NIFTY"):
            min_range, max_range = 80, 350
        elif instrument.upper() in ("FINNIFTY", "FIN NIFTY", "NIFTY FIN SERVICE"):
            min_range, max_range = 30, 120
        else:  # NIFTY
            min_range, max_range = 30, 120

        if orb_range < min_range:
            assessment = "narrow"
            tradeable = False  # Too narrow = choppy
        elif orb_range > max_range:
            assessment = "wide"
            tradeable = False  # Already moved too much
        else:
            assessment = "normal"
            tradeable = True

        return ORBAnalysis(
            orb_high=round(h, 2), orb_low=round(l, 2),
            orb_range=round(orb_range, 2), orb_range_pct=round(orb_range_pct, 2),
            assessment=assessment, tradeable=tradeable,
        )


@dataclass
class CPRAnalysis:
    """Central Pivot Range for the day."""
    pivot: float
    bc: float  # bottom central pivot
    tc: float  # top central pivot
    r1: float
    r2: float
    s1: float
    s2: float
    cpr_width_pct: float
    is_narrow: bool  # narrow CPR = likely trending day

    @staticmethod
    def from_prev_day(prev_high: float, prev_low: float, prev_close: float, spot: float) -> "CPRAnalysis":
        pivot = (prev_high + prev_low + prev_close) / 3
        bc = (prev_high + prev_low) / 2
        tc = 2 * pivot - bc
        # Ensure tc > bc
        if tc < bc:
            tc, bc = bc, tc
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        cpr_width = abs(tc - bc)
        cpr_width_pct = (cpr_width / spot * 100) if spot > 0 else 0
        # CPR < 0.3% of spot is considered narrow
        is_narrow = cpr_width_pct < 0.3

        return CPRAnalysis(
            pivot=round(pivot, 2), bc=round(bc, 2), tc=round(tc, 2),
            r1=round(r1, 2), r2=round(r2, 2),
            s1=round(s1, 2), s2=round(s2, 2),
            cpr_width_pct=round(cpr_width_pct, 3),
            is_narrow=is_narrow,
        )


# ══════════════════════════════════════════════════════════════════════════
#  DAY TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════


DAY_TYPES = ("trending_up", "trending_down", "sideways", "volatile", "expiry")


@dataclass
class DayClassification:
    """Classified trading day with reasoning."""
    day_type: str  # trending_up / trending_down / sideways / volatile / expiry
    direction: str  # bullish / bearish / neutral
    confidence: float  # 0-1
    factors: list[str]  # human-readable reasons
    vix: VIXRegime
    pcr: PCRRegime
    gap: Optional[GapAnalysis] = None
    orb: Optional[ORBAnalysis] = None
    cpr: Optional[CPRAnalysis] = None
    adx_value: Optional[float] = None
    is_weekly_expiry: bool = False
    is_monthly_expiry: bool = False
    position_size_multiplier: float = 1.0

    @property
    def summary(self) -> str:
        type_labels = {
            "trending_up": "Trending Up (Bullish)",
            "trending_down": "Trending Down (Bearish)",
            "sideways": "Sideways / Range-Bound",
            "volatile": "Volatile / Choppy",
            "expiry": "Expiry Day",
        }
        label = type_labels.get(self.day_type, self.day_type)
        return f"{label} ({self.confidence:.0%} confidence)"


def classify_day(
    vix_value: float,
    pcr_value: float,
    adx_value: Optional[float] = None,
    gap_analysis: Optional[GapAnalysis] = None,
    orb_analysis: Optional[ORBAnalysis] = None,
    cpr_analysis: Optional[CPRAnalysis] = None,
    today: Optional[date] = None,
    day_range_pct: Optional[float] = None,
    day_change_pct: Optional[float] = None,
    rsi_value: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> DayClassification:
    """
    Classify the trading day using PRICE-ACTION-FIRST approach.

    Design principles:
    1. Actual price action (change%, range%, directional efficiency) is ground truth
    2. Direction is explicit — trending_up vs trending_down
    3. Indicators (ADX, VIX, PCR) provide context, NOT classification override
    4. Directional efficiency = |change| / range — distinguishes trending from volatile

    Categories:
    - trending_up:  Market moved decisively higher
    - trending_down: Market moved decisively lower
    - sideways:     Small range, small change — nothing happened
    - volatile:     Large range but no clear direction (whipsaw / choppy)
    - expiry:       Weekly/monthly expiry — theta decay dominates
    """
    if today is None:
        today = date.today()

    vix = VIXRegime.classify(vix_value)
    pcr = PCRRegime.classify(pcr_value)

    # Check if it's expiry day
    weekday = today.weekday()  # 0=Mon ... 4=Fri
    is_thursday = weekday == 3
    import calendar
    _, last_day = calendar.monthrange(today.year, today.month)
    last_date = date(today.year, today.month, last_day)
    while last_date.weekday() != 3:
        last_day -= 1
        last_date = date(today.year, today.month, last_day)
    is_monthly_expiry = today == last_date
    is_weekly_expiry = is_thursday and not is_monthly_expiry

    factors = []

    # ── STEP 1: Compute price-action metrics (the GROUND TRUTH) ──
    abs_change = abs(day_change_pct) if day_change_pct is not None else None
    range_pct = day_range_pct  # (high-low)/close * 100

    # Directional efficiency: how much of the range was directional
    # 1.0 = pure trend (open-to-close equals the range)
    # 0.0 = pure reversal (close == open despite range)
    if abs_change is not None and range_pct and range_pct > 0:
        efficiency = abs_change / range_pct
    else:
        efficiency = None

    # ── STEP 2: Classify from price action ──
    day_type = None
    direction = "neutral"
    confidence = 0.50

    has_price_data = abs_change is not None and range_pct is not None

    if has_price_data:
        # Determine direction from change
        if day_change_pct > 0.1:
            direction = "bullish"
        elif day_change_pct < -0.1:
            direction = "bearish"

        # Tier 1: Extreme moves — unambiguous classification
        if abs_change >= 2.0:
            day_type = "trending_up" if direction == "bullish" else "trending_down"
            confidence = min(0.70 + abs_change * 0.07, 0.97)
            factors.append(f"Day change {day_change_pct:+.2f}% — extreme directional move")

        # Tier 2: Strong moves — directional day
        elif abs_change >= 1.0:
            day_type = "trending_up" if direction == "bullish" else "trending_down"
            confidence = 0.60 + abs_change * 0.10
            factors.append(f"Day change {day_change_pct:+.2f}% — clear directional day")

        # Tier 3: Wide range but small net change = volatile/choppy
        elif range_pct >= 1.8 and efficiency is not None and efficiency < 0.35:
            day_type = "volatile"
            confidence = 0.55 + range_pct * 0.08
            factors.append(f"Range {range_pct:.2f}% but change only {day_change_pct:+.2f}% — "
                           f"choppy/whipsaw (efficiency {efficiency:.0%})")

        # Tier 4: Moderate move with good efficiency = mild trend
        elif abs_change >= 0.5 and efficiency is not None and efficiency >= 0.45:
            day_type = "trending_up" if direction == "bullish" else "trending_down"
            confidence = 0.55 + abs_change * 0.08
            factors.append(f"Day change {day_change_pct:+.2f}%, efficiency {efficiency:.0%} — "
                           f"moderate but directional")

        # Tier 5: Wide range with moderate efficiency = volatile
        elif range_pct >= 1.3 and efficiency is not None and efficiency < 0.40:
            day_type = "volatile"
            confidence = 0.52 + range_pct * 0.06
            factors.append(f"Range {range_pct:.2f}%, low efficiency {efficiency:.0%} — choppy day")

        # Tier 6: Very quiet day
        elif abs_change < 0.3 and range_pct < 0.7:
            day_type = "sideways"
            confidence = 0.70
            factors.append(f"Change {day_change_pct:+.2f}%, range {range_pct:.2f}% — very quiet day")

        # Tier 7: Mild day — small change and modest range
        elif abs_change < 0.5 and range_pct < 1.1:
            day_type = "sideways"
            confidence = 0.58
            factors.append(f"Change {day_change_pct:+.2f}%, range {range_pct:.2f}% — sideways")

        # Tier 8: Borderline — moderate change, narrow range, still directional
        elif abs_change >= 0.5:
            day_type = "trending_up" if direction == "bullish" else "trending_down"
            confidence = 0.50 + abs_change * 0.06
            factors.append(f"Day change {day_change_pct:+.2f}% — borderline directional")

        # Tier 9: Default to sideways for anything left
        else:
            day_type = "sideways"
            confidence = 0.52
            factors.append(f"Change {day_change_pct:+.2f}%, range {range_pct:.2f}% — "
                           f"no strong signal, defaulting sideways")

    # ── STEP 3: Indicators as CONTEXT (add to factors, minor confidence adjustments) ──

    # VIX context
    if vix.level == "low":
        factors.append(f"VIX {vix_value:.1f} (low) — premiums cheap, favours buying")
        if day_type == "sideways":
            confidence = min(confidence + 0.05, 0.97)  # confirms sideways
    elif vix.level == "normal":
        factors.append(f"VIX {vix_value:.1f} (normal) — balanced premiums")
    elif vix.level == "high":
        factors.append(f"VIX {vix_value:.1f} (high) — elevated fear, premiums expensive")
        if day_type and "trending" in day_type:
            confidence = min(confidence + 0.03, 0.97)
    elif vix.level == "extreme":
        factors.append(f"VIX {vix_value:.1f} (extreme) — panic mode, hedge everything")
        if day_type == "volatile":
            confidence = min(confidence + 0.05, 0.97)

    # ADX context (trend strength over last 14 days — NOT today's action)
    if adx_value is not None:
        if adx_value > 30:
            factors.append(f"ADX {adx_value:.0f} (>30) — strong recent trend supports directional bias")
            if day_type and "trending" in day_type:
                confidence = min(confidence + 0.05, 0.97)
        elif adx_value > 25:
            factors.append(f"ADX {adx_value:.0f} (>25) — moderate trend in background")
            if day_type and "trending" in day_type:
                confidence = min(confidence + 0.02, 0.97)
        elif adx_value < 18:
            factors.append(f"ADX {adx_value:.0f} (<18) — no recent trend, range regime")
            if day_type == "sideways":
                confidence = min(confidence + 0.05, 0.97)
        else:
            factors.append(f"ADX {adx_value:.0f} — borderline trend strength")

    # Gap context
    if gap_analysis:
        abs_gap = abs(gap_analysis.gap_pct)
        if abs_gap >= 1.0:
            factors.append(f"Gap {gap_analysis.direction} {gap_analysis.gap_pct:+.1f}% — "
                           f"large gap, watch for gap fill vs continuation")
        elif abs_gap >= 0.3:
            factors.append(f"Gap {gap_analysis.direction} {gap_analysis.gap_pct:+.1f}% — "
                           f"moderate gap")
        else:
            factors.append("Flat open — no gap bias")

    # ORB context
    if orb_analysis:
        if orb_analysis.assessment == "wide":
            factors.append(f"ORB {orb_analysis.orb_range:.0f}pts (wide) — confirms range expansion")
        elif orb_analysis.assessment == "narrow":
            factors.append(f"ORB {orb_analysis.orb_range:.0f}pts (narrow) — tight opening, breakout possible later")
        else:
            factors.append(f"ORB {orb_analysis.orb_range:.0f}pts (normal)")

    # CPR context
    if cpr_analysis:
        if cpr_analysis.is_narrow:
            factors.append(f"Narrow CPR ({cpr_analysis.cpr_width_pct:.2f}%) — potential breakout")
        else:
            factors.append(f"CPR width {cpr_analysis.cpr_width_pct:.2f}%")

    # PCR context
    if pcr_value > 0:
        if pcr_value > 1.3:
            factors.append(f"PCR {pcr_value:.2f} (>1.3) — heavy put writing, bullish support below")
        elif pcr_value < 0.7:
            factors.append(f"PCR {pcr_value:.2f} (<0.7) — call-heavy, potential resistance above")
        else:
            factors.append(f"PCR {pcr_value:.2f} — neutral positioning")

    # RSI context
    if rsi_value is not None:
        if rsi_value > 75:
            factors.append(f"RSI {rsi_value:.0f} — overbought, watch for pullback")
        elif rsi_value < 25:
            factors.append(f"RSI {rsi_value:.0f} — oversold, watch for bounce")
        elif 40 <= rsi_value <= 60:
            factors.append(f"RSI {rsi_value:.0f} — neutral momentum")

    # ATR context
    if atr_pct is not None:
        if atr_pct >= 2.0:
            factors.append(f"ATR {atr_pct:.1f}% — high-volatility regime")
        elif atr_pct >= 1.2:
            factors.append(f"ATR {atr_pct:.1f}% — elevated volatility")
        elif atr_pct < 0.6:
            factors.append(f"ATR {atr_pct:.1f}% — low volatility regime")

    # ── STEP 4: Fallback when no price data (pre-market / live) ──
    if day_type is None:
        # No OHLC yet — use indicators to make a best guess
        day_type, direction, confidence = _classify_from_indicators_only(
            vix=vix, pcr_value=pcr_value, adx_value=adx_value,
            gap_analysis=gap_analysis, cpr_analysis=cpr_analysis,
        )
        factors.insert(0, "No day OHLC yet — classification from indicators only (less reliable)")

    # ── STEP 5: Expiry override — only if actually expiry day ──
    if is_monthly_expiry:
        # Monthly expiry overrides unless there's an extreme move
        if abs_change is None or abs_change < 2.0:
            day_type = "expiry"
            confidence = max(confidence, 0.70)
        factors.append("Monthly expiry day — max theta decay + gamma risk")
    elif is_weekly_expiry:
        if abs_change is None or abs_change < 1.5:
            day_type = "expiry"
            confidence = max(confidence, 0.60)
        factors.append("Weekly expiry day — elevated theta decay")

    # ── STEP 6: Clamp confidence ──
    confidence = max(0.50, min(0.97, confidence))

    # Position size multiplier
    pos_mult = vix.position_size_multiplier
    if day_type == "volatile":
        pos_mult = min(pos_mult, 0.5)
    if is_monthly_expiry:
        pos_mult = min(pos_mult, 0.75)

    return DayClassification(
        day_type=day_type,
        direction=direction,
        confidence=confidence,
        factors=factors,
        vix=vix,
        pcr=pcr,
        gap=gap_analysis,
        orb=orb_analysis,
        cpr=cpr_analysis,
        adx_value=adx_value,
        is_weekly_expiry=is_weekly_expiry,
        is_monthly_expiry=is_monthly_expiry,
        position_size_multiplier=pos_mult,
    )


def _classify_from_indicators_only(
    vix: VIXRegime,
    pcr_value: float,
    adx_value: Optional[float],
    gap_analysis: Optional[GapAnalysis],
    cpr_analysis: Optional[CPRAnalysis],
) -> tuple[str, str, float]:
    """Fallback when no day OHLC data is available (pre-market).

    Returns (day_type, direction, confidence).
    Less reliable than price-action classification.
    """
    # Simple scoring for fallback
    trend_score = 0.0
    sideways_score = 0.0
    volatile_score = 0.0
    direction = "neutral"

    if vix.level in ("high", "extreme"):
        volatile_score += 3.0
    elif vix.level == "low":
        sideways_score += 2.0
    else:
        trend_score += 1.0
        sideways_score += 1.0

    if adx_value is not None:
        if adx_value > 25:
            trend_score += 2.5
        elif adx_value < 18:
            sideways_score += 2.5

    if gap_analysis:
        abs_gap = abs(gap_analysis.gap_pct)
        if abs_gap >= 1.0:
            trend_score += 3.0
            volatile_score += 2.0
            direction = "bullish" if gap_analysis.gap_pct > 0 else "bearish"
        elif abs_gap >= 0.3:
            trend_score += 1.5
            direction = "bullish" if gap_analysis.gap_pct > 0 else "bearish"
        else:
            sideways_score += 1.0

    if cpr_analysis:
        if cpr_analysis.is_narrow:
            trend_score += 1.5
        else:
            sideways_score += 0.5

    if pcr_value > 0:
        if pcr_value > 1.3:
            direction = "bullish"
        elif pcr_value < 0.7:
            direction = "bearish"

    best = max(trend_score, sideways_score, volatile_score)
    if best == volatile_score and volatile_score > 0:
        day_type = "volatile"
    elif best == trend_score and trend_score > 0:
        day_type = "trending_up" if direction == "bullish" else "trending_down"
    elif best == sideways_score and sideways_score > 0:
        day_type = "sideways"
    else:
        day_type = "sideways"

    # Low confidence for indicator-only classification
    total = trend_score + sideways_score + volatile_score
    confidence = (best / total * 0.7) if total > 0 else 0.50
    confidence = max(0.50, min(0.65, confidence))  # cap at 65% for indicator-only

    return day_type, direction, confidence


# ══════════════════════════════════════════════════════════════════════════
#  STRATEGY RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyRecommendation:
    """A recommended strategy with rationale."""
    name: str
    strategy_type: str  # directional_buying / premium_selling / hedged
    description: str
    suitability: str  # why it fits the current day
    risk_level: str  # low / medium / high
    entry_window: str
    key_levels: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)  # can be fed to CustomStrategy


# Strategy database — maps day types to appropriate strategies
STRATEGY_DATABASE = {
    "trending_up": [
        StrategyRecommendation(
            name="Bull Call Spread",
            strategy_type="directional_buying",
            description="Buy ATM Call + Sell OTM Call for defined-risk bullish trade",
            suitability="Market trending up — defined risk, lower cost than naked long",
            risk_level="low",
            entry_window="09:20-14:00 IST",
            config={
                "name": "Bull Call Spread",
                "conditions": [
                    {"type": "price_vs_ema", "params": {"period": 20, "position": "above"}},
                    {"type": "vwap", "params": {"position": "above"}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "BUY", "qty": 1},
                    {"template": "otm_call_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="OTM Call Momentum Scalp",
            strategy_type="directional_buying",
            description="Buy slightly OTM calls on strong upside breakout for quick 10-30% returns",
            suitability="Strong bullish trend — momentum carries OTM calls higher",
            risk_level="medium",
            entry_window="09:20-11:30 IST",
            config={
                "name": "OTM Call Scalp",
                "conditions": [
                    {"type": "supertrend", "params": {"period": 10, "multiplier": 3.0, "signal": "bullish"}},
                    {"type": "vwap", "params": {"position": "above"}},
                ],
                "condition_logic": "AND",
                "legs": [{"template": "otm_call_half", "action": "BUY", "qty": 1}],
            },
        ),
        StrategyRecommendation(
            name="Opening Range Breakout (Long)",
            strategy_type="directional_buying",
            description="Trade upside breakout of the first 15-minute candle range",
            suitability="Gap-up or strong open with bullish follow-through",
            risk_level="medium",
            entry_window="09:30-10:00 IST (after ORB forms)",
            config={
                "name": "ORB Long",
                "conditions": [
                    {"type": "gap", "params": {"direction": "up", "min_pct": 0.3}},
                    {"type": "consecutive_candles", "params": {"color": "green", "count": 2}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "BUY", "qty": 1},
                    {"template": "otm_call_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
    ],
    "trending_down": [
        StrategyRecommendation(
            name="Bear Put Spread",
            strategy_type="directional_buying",
            description="Buy ATM Put + Sell OTM Put for defined-risk bearish trade",
            suitability="Market trending down — defined risk, profits from continued decline",
            risk_level="low",
            entry_window="09:20-14:00 IST",
            config={
                "name": "Bear Put Spread",
                "conditions": [
                    {"type": "price_vs_ema", "params": {"period": 20, "position": "below"}},
                    {"type": "vwap", "params": {"position": "below"}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_put", "action": "BUY", "qty": 1},
                    {"template": "otm_put_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="OTM Put Momentum Scalp",
            strategy_type="directional_buying",
            description="Buy slightly OTM puts on strong downside breakdown for quick 10-30% returns",
            suitability="Strong bearish trend — panic selling makes OTM puts surge",
            risk_level="medium",
            entry_window="09:20-11:30 IST",
            config={
                "name": "OTM Put Scalp",
                "conditions": [
                    {"type": "supertrend", "params": {"period": 10, "multiplier": 3.0, "signal": "bearish"}},
                    {"type": "vwap", "params": {"position": "below"}},
                ],
                "condition_logic": "AND",
                "legs": [{"template": "otm_put_half", "action": "BUY", "qty": 1}],
            },
        ),
        StrategyRecommendation(
            name="Opening Range Breakdown (Short)",
            strategy_type="directional_buying",
            description="Trade downside breakdown of the first 15-minute candle range",
            suitability="Gap-down or weak open with bearish follow-through",
            risk_level="medium",
            entry_window="09:30-10:00 IST (after ORB forms)",
            config={
                "name": "ORB Short",
                "conditions": [
                    {"type": "gap", "params": {"direction": "down", "min_pct": 0.3}},
                    {"type": "consecutive_candles", "params": {"color": "red", "count": 2}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_put", "action": "BUY", "qty": 1},
                    {"template": "otm_put_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
    ],
    "sideways": [
        StrategyRecommendation(
            name="ATM Short Straddle",
            strategy_type="premium_selling",
            description="Sell ATM Call + ATM Put to collect premium in sideways market",
            suitability="Low range, low change — premium melts as market stays flat",
            risk_level="high",
            entry_window="09:20-09:35 IST",
            config={
                "name": "ATM Short Straddle",
                "conditions": [
                    {"type": "trend_strength", "params": {"period": 14, "operator": "<", "value": 20}},
                    {"type": "bollinger", "params": {"period": 20, "num_std": 2.0, "position": "within"}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "SELL", "qty": 1},
                    {"template": "atm_put", "action": "SELL", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="Intraday Iron Condor",
            strategy_type="premium_selling_hedged",
            description="Sell OTM Call + Put spreads for defined-risk premium collection",
            suitability="Sideways market — iron condor profits if market stays in range",
            risk_level="medium",
            entry_window="09:25-10:30 IST",
            config={
                "name": "Iron Condor",
                "conditions": [
                    {"type": "trend_strength", "params": {"period": 14, "operator": "<", "value": 22}},
                    {"type": "atr_rank", "params": {"period": 14, "operator": "<", "value": 1.5}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "otm_call_1", "action": "SELL", "qty": 1},
                    {"template": "otm_call_2", "action": "BUY", "qty": 1},
                    {"template": "otm_put_1", "action": "SELL", "qty": 1},
                    {"template": "otm_put_2", "action": "BUY", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="VWAP Mean Reversion",
            strategy_type="directional_buying",
            description="Buy options when price deviates from VWAP and shows reversal",
            suitability="Sideways market reverts to mean — buy dips, sell rips around VWAP",
            risk_level="medium",
            entry_window="10:00-14:00 IST",
            config={
                "name": "VWAP Reversion",
                "conditions": [
                    {"type": "vwap", "params": {"position": "below"}},
                    {"type": "rsi", "params": {"period": 14, "operator": "<", "value": 35}},
                ],
                "condition_logic": "AND",
                "legs": [{"template": "atm_call", "action": "BUY", "qty": 1}],
            },
        ),
    ],
    "volatile": [
        StrategyRecommendation(
            name="Long Straddle",
            strategy_type="directional_buying",
            description="Buy ATM Call + ATM Put to profit from big move in either direction",
            suitability="High volatility, wide range — big moves expected, direction uncertain",
            risk_level="medium",
            entry_window="09:20-10:00 IST",
            config={
                "name": "Long Straddle",
                "conditions": [
                    {"type": "atr_rank", "params": {"period": 14, "operator": ">", "value": 2.0}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "BUY", "qty": 1},
                    {"template": "atm_put", "action": "BUY", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="ORB with Wide Stops",
            strategy_type="directional_buying",
            description="Trade ORB breakout with wider stops to accommodate volatile swings",
            suitability="High VIX = big moves. Capture the initial directional thrust.",
            risk_level="high",
            entry_window="09:30-10:00 IST",
            config={
                "name": "Volatile ORB",
                "conditions": [
                    {"type": "gap", "params": {"direction": "up", "min_pct": 0.5}},
                    {"type": "consecutive_candles", "params": {"color": "green", "count": 2}},
                ],
                "condition_logic": "AND",
                "legs": [{"template": "otm_call_half", "action": "BUY", "qty": 1}],
            },
        ),
    ],
    "expiry": [
        StrategyRecommendation(
            name="Expiry Day Theta Decay Strangle",
            strategy_type="premium_selling",
            description="Sell 1-SD wide strangle to exploit rapid theta decay on expiry",
            suitability="Weekly expiry = max theta decay. Premium melts fastest on this day.",
            risk_level="high",
            entry_window="09:25-10:00 IST",
            config={
                "name": "Expiry Strangle Sell",
                "conditions": [
                    {"type": "day_of_week", "params": {"days": ["thu"]}},
                    {"type": "trend_strength", "params": {"period": 14, "operator": "<", "value": 25}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "otm_call_1", "action": "SELL", "qty": 1},
                    {"template": "otm_put_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="Expiry ATM Short Straddle",
            strategy_type="premium_selling",
            description="Sell ATM straddle on expiry for maximum theta harvest",
            suitability="ATM options lose time value fastest on expiry day",
            risk_level="high",
            entry_window="09:20-09:35 IST",
            config={
                "name": "Expiry Straddle Sell",
                "conditions": [
                    {"type": "day_of_week", "params": {"days": ["thu"]}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "SELL", "qty": 1},
                    {"template": "atm_put", "action": "SELL", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="Expiry Iron Butterfly",
            strategy_type="premium_selling_hedged",
            description="Iron butterfly on expiry — sell ATM, buy OTM wings for defined risk",
            suitability="Max theta with capped risk on expiry",
            risk_level="medium",
            entry_window="09:25-10:00 IST",
            config={
                "name": "Expiry Iron Butterfly",
                "conditions": [
                    {"type": "day_of_week", "params": {"days": ["thu"]}},
                    {"type": "bollinger", "params": {"period": 20, "num_std": 2.0, "position": "within"}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "SELL", "qty": 1},
                    {"template": "atm_put", "action": "SELL", "qty": 1},
                    {"template": "otm_call_1", "action": "BUY", "qty": 1},
                    {"template": "otm_put_1", "action": "BUY", "qty": 1},
                ],
            },
        ),
    ],
}


def recommend_strategies(day: DayClassification) -> list[StrategyRecommendation]:
    """Get strategy recommendations for the classified day."""
    primary = STRATEGY_DATABASE.get(day.day_type, [])
    result = list(primary)

    # Also include one strategy from a complementary type if confidence isn't high
    if day.confidence < 0.65:
        # Map to a sensible secondary type based on the primary
        secondary_map = {
            "trending_up": "sideways",      # if unsure, hedge with sideways strats
            "trending_down": "volatile",     # bearish + unsure → volatile strats
            "sideways": "trending_up",       # if unsure sideways, add a directional
            "volatile": "trending_down",     # volatile + unsure → bearish protection
            "expiry": "sideways",
        }
        secondary_type = secondary_map.get(day.day_type, "sideways")
        secondary = STRATEGY_DATABASE.get(secondary_type, [])
        if secondary:
            result.append(secondary[0])
    return result


# ══════════════════════════════════════════════════════════════════════════
#  MARKET HOURS & TIME WINDOWS
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class MarketTimeContext:
    """Current market time context with actionable guidance."""
    current_time: dtime
    phase: str  # pre_market / opening_avoid / first_trade / morning_prime / lunch_lull / afternoon / exit_zone / closed
    can_enter: bool
    can_exit: bool
    guidance: str

    @staticmethod
    def now() -> "MarketTimeContext":
        now = datetime.now(IST).time()
        if now < dtime(9, 0):
            return MarketTimeContext(now, "pre_market", False, False, "Market not open. Complete pre-market analysis.")
        if now < dtime(9, 15):
            return MarketTimeContext(now, "pre_market", False, False, "Pre-market session. Analyse global cues, VIX, OI.")
        if now < dtime(9, 20):
            return MarketTimeContext(now, "opening_avoid", False, False, "Avoid! First 5 minutes have high spreads and erratic fills.")
        if now < dtime(9, 30):
            return MarketTimeContext(now, "first_trade", True, True, "First trade window. ORB forming. Enter selling strategies now.")
        if now < dtime(11, 30):
            return MarketTimeContext(now, "morning_prime", True, True, "Prime trading hours. Best setups form now.")
        if now < dtime(12, 30):
            return MarketTimeContext(now, "morning_prime", True, True, "Late morning — still good for entries with strong conviction.")
        if now < dtime(13, 30):
            return MarketTimeContext(now, "lunch_lull", False, True, "Lunch hours — low volume. Avoid new entries. Exit if targets met.")
        if now < dtime(14, 30):
            return MarketTimeContext(now, "afternoon", True, True, "Afternoon session. Last window for new entries.")
        if now < dtime(15, 15):
            return MarketTimeContext(now, "exit_zone", False, True, "Exit zone! Square off positions. No new entries.")
        if now <= dtime(15, 30):
            return MarketTimeContext(now, "exit_zone", False, True, "Final minutes — mandatory exit. Watch slippage.")
        return MarketTimeContext(now, "closed", False, False, "Market closed. Review trades and journal.")


# ══════════════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT CONTEXT
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class RiskContext:
    """Per-trade and daily risk parameters."""
    capital: float
    max_risk_per_trade_pct: float = 1.5
    max_capital_per_trade_pct: float = 10.0
    max_daily_loss_pct: float = 2.0
    max_daily_trades: int = 6
    consecutive_loss_limit: int = 3
    slippage_pct: float = 0.5
    position_size_multiplier: float = 1.0

    @property
    def max_risk_per_trade(self) -> float:
        return self.capital * (self.max_risk_per_trade_pct / 100) * self.position_size_multiplier

    @property
    def max_daily_loss(self) -> float:
        return self.capital * (self.max_daily_loss_pct / 100)

    @property
    def max_capital_per_trade(self) -> float:
        return self.capital * (self.max_capital_per_trade_pct / 100) * self.position_size_multiplier

    def max_lots(self, lot_size: int, option_premium: float) -> int:
        """Calculate maximum lots based on risk limits."""
        if option_premium <= 0:
            return 0
        capital_lots = int(self.max_capital_per_trade / (option_premium * lot_size))
        risk_lots = int(self.max_risk_per_trade / (option_premium * lot_size * 0.3))  # assume 30% SL
        return max(1, min(capital_lots, risk_lots, 4))  # cap at 4 lots
