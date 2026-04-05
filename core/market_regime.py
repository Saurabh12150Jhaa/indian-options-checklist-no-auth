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


DAY_TYPES = ("trending", "range_bound", "volatile", "expiry")


@dataclass
class DayClassification:
    """Classified trading day with reasoning."""
    day_type: str  # trending / range_bound / volatile / expiry
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
            "trending": "Trending Day",
            "range_bound": "Range-Bound Day",
            "volatile": "Volatile / Event Day",
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
    Classify the trading day based on available market data.

    Uses a weighted scoring system where:
    - Each factor votes for one or more day types
    - Confidence = best_score / (best_score + second_best_score)
      so it measures *separation* between the top two, not diluted by all four
    - Additional hard data (day range, % change) can override weaker signals

    Args:
        vix_value: India VIX (0 if unavailable)
        pcr_value: Put-Call Ratio (0 if unavailable — will be ignored)
        adx_value: ADX(14) trend strength
        gap_analysis: Opening gap classification
        orb_analysis: Opening Range Breakout analysis
        cpr_analysis: Central Pivot Range
        today: Date to classify (defaults to today)
        day_range_pct: (High - Low) / Close * 100 — actual intraday range
        day_change_pct: (Close - PrevClose) / PrevClose * 100 — daily return
        rsi_value: RSI(14) — momentum/oversold/overbought
        atr_pct: ATR(14) as % of close — recent volatility context
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

    # ── Scoring: each factor adds to one or more day types ──
    scores = {"trending": 0.0, "range_bound": 0.0, "volatile": 0.0, "expiry": 0.0}
    factors = []

    # Factor 1: VIX (weight: 2-3)
    if vix.level == "low":
        scores["range_bound"] += 2.0
        factors.append(f"VIX {vix_value:.1f} (low) — favours range-bound / premium selling")
    elif vix.level == "normal":
        scores["trending"] += 1.0
        scores["range_bound"] += 1.0
        factors.append(f"VIX {vix_value:.1f} (normal) — both directional and neutral viable")
    elif vix.level == "high":
        scores["volatile"] += 2.5
        scores["trending"] += 1.5
        factors.append(f"VIX {vix_value:.1f} (high) — volatility elevated, directional moves likely")
    elif vix.level == "extreme":
        scores["volatile"] += 4.0
        scores["trending"] += 1.0
        factors.append(f"VIX {vix_value:.1f} (extreme) — very high fear, hedge everything")

    # Factor 2: ADX (weight: 2.5-3)
    if adx_value is not None:
        if adx_value > 30:
            scores["trending"] += 3.0
            factors.append(f"ADX {adx_value:.0f} (>30) — very strong trend")
        elif adx_value > 25:
            scores["trending"] += 2.5
            factors.append(f"ADX {adx_value:.0f} (>25) — strong trend detected")
        elif adx_value < 18:
            scores["range_bound"] += 3.0
            factors.append(f"ADX {adx_value:.0f} (<18) — no trend, range-bound")
        elif adx_value < 22:
            scores["range_bound"] += 1.5
            factors.append(f"ADX {adx_value:.0f} — weak trend, likely range-bound")
        else:
            scores["trending"] += 0.5
            scores["range_bound"] += 0.5
            factors.append(f"ADX {adx_value:.0f} — borderline trend strength")

    # Factor 3: Gap (weight: 2-4 — gaps are very strong signals)
    if gap_analysis:
        abs_gap = abs(gap_analysis.gap_pct)
        if abs_gap >= 2.0:
            scores["volatile"] += 3.0
            scores["trending"] += 3.0
            factors.append(f"Gap {gap_analysis.direction} {gap_analysis.gap_pct:+.1f}% — "
                           f"massive gap, highly directional/volatile open")
        elif abs_gap >= 1.0:
            scores["trending"] += 2.5
            scores["volatile"] += 1.5
            factors.append(f"Gap {gap_analysis.direction} {gap_analysis.gap_pct:+.1f}% — "
                           f"large gap, strong directional open")
        elif abs_gap >= 0.5:
            scores["trending"] += 1.5
            factors.append(f"Gap {gap_analysis.direction} {gap_analysis.gap_pct:+.1f}% — "
                           f"moderate gap")
        elif abs_gap < 0.2:
            scores["range_bound"] += 1.0
            factors.append("Flat open — no gap bias")

    # Factor 4: ORB
    if orb_analysis:
        if orb_analysis.assessment == "wide":
            scores["trending"] += 2.0
            factors.append(f"ORB range {orb_analysis.orb_range:.0f}pts (wide) — trending signal")
        elif orb_analysis.assessment == "narrow":
            scores["range_bound"] += 2.0
            factors.append(f"ORB range {orb_analysis.orb_range:.0f}pts (narrow) — consolidation")
        else:
            factors.append(f"ORB range {orb_analysis.orb_range:.0f}pts (normal)")

    # Factor 5: CPR (weight: 1.5 — confirmatory, should NOT override strong signals)
    if cpr_analysis:
        if cpr_analysis.is_narrow:
            scores["trending"] += 1.5
            factors.append(f"Narrow CPR ({cpr_analysis.cpr_width_pct:.2f}%) — trending signal")
        else:
            scores["range_bound"] += 0.5  # reduced from 1.0 — wide CPR is weak signal
            factors.append(f"CPR width {cpr_analysis.cpr_width_pct:.2f}%")

    # Factor 6: PCR (weight: 1-2 — only when data is available)
    if pcr_value > 0:
        if pcr_value > 1.3:
            scores["trending"] += 1.5  # heavy put writing = bullish support
            factors.append(f"PCR {pcr_value:.2f} (>1.3) — strong put writing, bullish underpinning")
        elif pcr_value < 0.7:
            scores["trending"] += 1.0  # low PCR can mean bearish trend
            scores["volatile"] += 0.5
            factors.append(f"PCR {pcr_value:.2f} (<0.7) — bearish, calls dominate")
        else:
            scores["range_bound"] += 0.5
            factors.append(f"PCR {pcr_value:.2f} — neutral")

    # Factor 7: Actual day range (weight: 3-5 — HARD DATA, strongest signal)
    if day_range_pct is not None:
        if day_range_pct >= 3.0:
            scores["volatile"] += 5.0
            scores["trending"] += 3.0
            factors.append(f"Day range {day_range_pct:.1f}% — extreme intraday swing")
        elif day_range_pct >= 2.0:
            scores["volatile"] += 3.0
            scores["trending"] += 2.0
            factors.append(f"Day range {day_range_pct:.1f}% — very wide range")
        elif day_range_pct >= 1.2:
            scores["trending"] += 2.5
            factors.append(f"Day range {day_range_pct:.1f}% — above-average movement")
        elif day_range_pct < 0.5:
            scores["range_bound"] += 3.0
            factors.append(f"Day range {day_range_pct:.1f}% — very narrow, range-bound")
        elif day_range_pct < 0.8:
            scores["range_bound"] += 1.5
            factors.append(f"Day range {day_range_pct:.1f}% — below-average movement")

    # Factor 8: Daily change (weight: 2-4 — HARD DATA)
    if day_change_pct is not None:
        abs_change = abs(day_change_pct)
        if abs_change >= 3.0:
            scores["volatile"] += 4.0
            scores["trending"] += 3.0
            factors.append(f"Day change {day_change_pct:+.1f}% — extreme move")
        elif abs_change >= 1.5:
            scores["trending"] += 3.0
            scores["volatile"] += 1.0
            factors.append(f"Day change {day_change_pct:+.1f}% — strong directional move")
        elif abs_change >= 0.8:
            scores["trending"] += 2.0
            factors.append(f"Day change {day_change_pct:+.1f}% — clear directional day")
        elif abs_change < 0.2:
            scores["range_bound"] += 2.0
            factors.append(f"Day change {day_change_pct:+.1f}% — unchanged, range-bound")

    # Factor 9: RSI (weight: 1 — supplementary)
    if rsi_value is not None:
        if rsi_value > 75:
            scores["trending"] += 1.0
            factors.append(f"RSI {rsi_value:.0f} — overbought, strong uptrend")
        elif rsi_value < 25:
            scores["trending"] += 1.0
            factors.append(f"RSI {rsi_value:.0f} — oversold, strong downtrend")
        elif 40 <= rsi_value <= 60:
            scores["range_bound"] += 0.5
            factors.append(f"RSI {rsi_value:.0f} — neutral")

    # Factor 10: ATR % (weight: 1-2 — recent volatility baseline)
    if atr_pct is not None:
        if atr_pct >= 2.0:
            scores["volatile"] += 2.0
            factors.append(f"ATR {atr_pct:.1f}% — market in high-volatility regime")
        elif atr_pct >= 1.2:
            scores["trending"] += 1.0
            factors.append(f"ATR {atr_pct:.1f}% — elevated volatility")
        elif atr_pct < 0.6:
            scores["range_bound"] += 1.0
            factors.append(f"ATR {atr_pct:.1f}% — low volatility regime")

    # Factor 11: Expiry (weight: 3-4 — can override but only IF it's actually expiry)
    if is_monthly_expiry:
        scores["expiry"] += 4.0
        factors.append("Monthly expiry day — max theta decay + gamma risk")
    elif is_weekly_expiry:
        scores["expiry"] += 3.0
        factors.append("Weekly expiry day — elevated theta decay")

    # ── Determine winner with proper confidence ──
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type = sorted_types[0][0]
    best_score = sorted_types[0][1]
    second_type = sorted_types[1][0] if len(sorted_types) > 1 else ""
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0
    third_score = sorted_types[2][1] if len(sorted_types) > 2 else 0.0

    # Confidence = how much the winner dominates the runner-up
    if best_score + second_score > 0:
        confidence = best_score / (best_score + second_score)
    else:
        confidence = 0.5

    # Synergy boost: "trending" and "volatile" are NOT contradictory — a crash is
    # both trending and volatile. When these two dominate and the rest is negligible,
    # the classification is actually very certain (just debating which label).
    # Boost confidence using margin over the THIRD place instead.
    synergy_pairs = {("trending", "volatile"), ("volatile", "trending")}
    if (best_type, second_type) in synergy_pairs and best_score > 5 and second_score > 5:
        # Both trending and volatile are strong — use margin over 3rd place
        if best_score + second_score > 0 and third_score >= 0:
            # Confidence = how much the top TWO dominate the rest
            combined = best_score + second_score
            synergy_conf = combined / (combined + third_score) if (combined + third_score) > 0 else 0.8
            # Use the higher of the two confidence measures
            confidence = max(confidence, synergy_conf * 0.95)  # scale slightly under 1.0

    # Floor confidence at 50% (random) and cap at 97%
    confidence = max(0.5, min(0.97, confidence))

    # Position size multiplier
    pos_mult = vix.position_size_multiplier
    if best_type == "volatile":
        pos_mult = min(pos_mult, 0.5)
    if is_monthly_expiry:
        pos_mult = min(pos_mult, 0.75)

    return DayClassification(
        day_type=best_type,
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
    "trending": [
        StrategyRecommendation(
            name="OTM Momentum Scalp",
            strategy_type="directional_buying",
            description="Buy slightly OTM options on strong momentum breakouts for quick 10-30% returns",
            suitability="Trending market allows directional moves to sustain",
            risk_level="medium",
            entry_window="09:20-11:30 IST",
            config={
                "name": "OTM Momentum Scalp",
                "conditions": [
                    {"type": "supertrend", "params": {"period": 10, "multiplier": 3.0, "signal": "bullish"}},
                    {"type": "vwap", "params": {"position": "above"}},
                ],
                "condition_logic": "AND",
                "legs": [{"template": "otm_call_half", "action": "BUY", "qty": 1}],
            },
        ),
        StrategyRecommendation(
            name="Opening Range Breakout (ORB)",
            strategy_type="directional_buying",
            description="Trade the breakout of the first 15-minute candle range",
            suitability="Strong directional open with follow-through potential",
            risk_level="medium",
            entry_window="09:30-10:00 IST (after ORB forms)",
            config={
                "name": "ORB Breakout",
                "conditions": [
                    {"type": "gap", "params": {"direction": "up", "min_pct": 0.3}},
                    {"type": "trend_strength", "params": {"period": 14, "operator": ">", "value": 22}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "BUY", "qty": 1},
                    {"template": "otm_call_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
        StrategyRecommendation(
            name="Trend-Following Bull/Bear Spread",
            strategy_type="directional_buying",
            description="Defined-risk directional spread in the direction of the trend",
            suitability="ADX confirms strong trend, spread limits risk",
            risk_level="low",
            entry_window="09:20-14:00 IST",
            config={
                "name": "Trend Bull Spread",
                "conditions": [
                    {"type": "price_vs_ema", "params": {"period": 20, "position": "above"}},
                    {"type": "trend_strength", "params": {"period": 14, "operator": ">", "value": 25}},
                ],
                "condition_logic": "AND",
                "legs": [
                    {"template": "atm_call", "action": "BUY", "qty": 1},
                    {"template": "otm_call_1", "action": "SELL", "qty": 1},
                ],
            },
        ),
    ],
    "range_bound": [
        StrategyRecommendation(
            name="ATM Short Straddle",
            strategy_type="premium_selling",
            description="Sell ATM Call + ATM Put to collect premium in sideways market",
            suitability="Low ADX and range-bound conditions maximise theta income",
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
            suitability="Range-bound with defined risk — iron condor profits if market stays in range",
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
            description="Buy options when price deviates significantly from VWAP and shows reversal",
            suitability="Range-bound days see mean reversion around VWAP",
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
        StrategyRecommendation(
            name="Hedged Straddle Buy",
            strategy_type="directional_buying",
            description="Buy ATM straddle to profit from expected big move in either direction",
            suitability="Extreme VIX or event day — big move expected but direction uncertain",
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

    # Also include one strategy from the secondary type if confidence isn't high
    if day.confidence < 0.65:
        # Use the actual scores from the classification rather than parsing factor strings
        # Factor strings contain hyphens ("range-bound") while keys use underscores ("range_bound")
        # so keyword matching is unreliable. Instead, infer the secondary type by looking at
        # which day types got mentioned most in the factors, mapping both forms.
        type_keywords = {
            "trending": ["trending", "trend", "directional", "momentum", "breakout"],
            "range_bound": ["range-bound", "range_bound", "sideways", "consolidation", "narrow"],
            "volatile": ["volatile", "volatility", "extreme", "fear", "wild", "swing"],
            "expiry": ["expiry", "theta", "gamma"],
        }
        scores = {t: 0 for t in DAY_TYPES}
        factor_text = " ".join(day.factors).lower()
        for day_type, keywords in type_keywords.items():
            for kw in keywords:
                scores[day_type] += factor_text.count(kw)
        # Exclude primary
        scores[day.day_type] = -1
        secondary_type = max(scores, key=scores.get)
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
