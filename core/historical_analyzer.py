"""
Historical Analysis Engine — run the complete analysis pipeline for any date.

Given a date and symbol, fetches REAL data from yfinance and bhavcopy cache,
computes all technical indicators, options analytics, day classification, and
strategy recommendations using actual market data.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import INDEX_CONFIG, INDIA_VIX_TICKER, BHAVCOPY_CACHE_DIR

logger = logging.getLogger(__name__)

# Module-level caches to avoid redundant network calls within a session
_ohlc_cache: dict[str, pd.DataFrame] = {}
_vix_cache: dict[str, pd.DataFrame] = {}


# ══════════════════════════════════════════════════════════════════════════
#  DATA FETCHERS — real historical data
# ══════════════════════════════════════════════════════════════════════════


def fetch_historical_ohlc(
    symbol: str, target_date: date, lookback_days: int = 60,
) -> Optional[pd.DataFrame]:
    """
    Fetch real daily OHLC from yfinance for the given symbol,
    covering `lookback_days` before `target_date` through `target_date`.
    Returns DataFrame with columns: date, open, high, low, close, volume.

    Results are cached in ``_ohlc_cache`` to avoid redundant network calls.
    Note: yfinance calls may take 2-10 s depending on network conditions.
    """
    try:
        cache_key = f"{symbol}_{target_date}_{lookback_days}"
        if cache_key in _ohlc_cache:
            return _ohlc_cache[cache_key]

        cfg = INDEX_CONFIG.get(symbol)
        if not cfg:
            logger.warning("Unknown symbol: %s", symbol)
            return None

        ticker = cfg["yf_ticker"]
        start = target_date - timedelta(days=lookback_days + 10)  # buffer for weekends/holidays
        end = target_date + timedelta(days=1)  # yfinance end is exclusive

        tk = yf.Ticker(ticker)
        df = tk.history(start=str(start), end=str(end))
        if df is None or df.empty:
            logger.warning("No yfinance data for %s from %s to %s", ticker, start, end)
            return None
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        result = df[keep].copy()
        # Filter to only dates <= target_date
        result = result[result["date"] <= target_date]
        if result.empty:
            return None

        _ohlc_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error("Failed to fetch OHLC for %s: %s", symbol, e)
        return None


def fetch_historical_vix(target_date: date, lookback_days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch historical India VIX values from yfinance.
    Returns DataFrame with columns: date, vix.

    Results are cached in ``_vix_cache`` to avoid redundant network calls.
    Note: yfinance calls may take 2-10 s depending on network conditions.
    """
    try:
        cache_key = f"vix_{target_date}_{lookback_days}"
        if cache_key in _vix_cache:
            return _vix_cache[cache_key]

        start = target_date - timedelta(days=lookback_days + 10)
        end = target_date + timedelta(days=1)

        tk = yf.Ticker(INDIA_VIX_TICKER)
        df = tk.history(start=str(start), end=str(end))
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        result = df[["date", "close"]].copy()
        result.columns = ["date", "vix"]
        result = result[result["date"] <= target_date]
        if result.empty:
            return None

        _vix_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error("Failed to fetch VIX history: %s", e)
        return None


def load_bhavcopy_for_date(
    target_date: date, symbol: str,
) -> Optional[pd.DataFrame]:
    """
    Load option chain data from cached bhavcopy for the given date and symbol.
    Searches the bhavcopy cache directory for files matching the date.
    Returns standardised DataFrame or None if not available.
    """
    cache_dir = BHAVCOPY_CACHE_DIR
    if not cache_dir.exists():
        return None

    try:
        from backtester.data_adapter import parse_bhavcopy_csv
    except ImportError:
        return None

    # Search for files matching the date — multiple patterns for robustness
    # Bhavcopy filenames: fo{DDMMMYYYY}bhav.csv.zip or various patterns
    date_str_patterns = [
        target_date.strftime("%d%b%Y").upper(),   # 01JAN2024
        target_date.strftime("%d%b%Y").lower(),   # 01jan2024
        target_date.strftime("%d%m%Y"),            # 01012024
        target_date.strftime("%Y%m%d"),            # 20240101
        target_date.strftime("%Y-%m-%d"),          # 2024-01-01
        target_date.strftime("%d-%m-%Y"),          # 01-01-2024
        target_date.strftime("%d_%m_%Y"),          # 01_01_2024
        target_date.strftime("%d-%b-%Y").upper(),  # 01-JAN-2024
    ]

    matching_files = []
    try:
        for f in sorted(cache_dir.iterdir()):
            if not f.is_file():
                continue
            fname = f.name.upper()
            for pattern in date_str_patterns:
                if pattern.upper() in fname:
                    matching_files.append(f)
                    break
    except OSError as e:
        logger.error("Error iterating bhavcopy cache dir %s: %s", cache_dir, e)
        return None

    if not matching_files:
        return None

    frames = []
    for fp in matching_files:
        try:
            df = parse_bhavcopy_csv(fp)
            if df is None or df.empty:
                continue
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df[df["date"] == target_date]
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning("Failed to parse bhavcopy file %s: %s", fp.name, e)
            continue

    if not frames:
        return None

    try:
        result = pd.concat(frames, ignore_index=True)
        return result if not result.empty else None
    except Exception as e:
        logger.error("Failed to concat bhavcopy frames: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATOR COMPUTATION — from real OHLC
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class TechnicalSnapshot:
    """All technical indicators for a specific date."""
    date: date
    spot: float
    prev_close: float
    day_open: float
    day_high: float
    day_low: float
    day_close: float
    volume: int = 0

    # EMAs
    ema_9: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None

    # RSI
    rsi_14: Optional[float] = None

    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # ADX
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width_pct: Optional[float] = None

    # Supertrend
    supertrend_direction: Optional[str] = None  # "bullish" / "bearish"

    # VWAP (approximated from daily data)
    vwap: Optional[float] = None

    # ATR
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None

    # Pivot Points
    pivot: Optional[float] = None
    r1: Optional[float] = None
    r2: Optional[float] = None
    s1: Optional[float] = None
    s2: Optional[float] = None

    # Gap
    gap_pct: Optional[float] = None
    gap_direction: Optional[str] = None


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float safely, returning *default* on NaN / error."""
    try:
        f = float(val)
        return default if pd.isna(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return default


def compute_technical_snapshot(ohlc: pd.DataFrame, target_date: date) -> Optional[TechnicalSnapshot]:
    """
    Compute all technical indicators from real OHLC data for the target date.
    ohlc must have columns: date, open, high, low, close, volume (sorted by date ascending).

    Each indicator block is independently guarded so one failure does not
    prevent the others from being computed.
    """
    if ohlc is None or ohlc.empty:
        return None

    try:
        # Ensure sorted
        ohlc = ohlc.sort_values("date").reset_index(drop=True)

        # Find the target date row
        target_rows = ohlc[ohlc["date"] == target_date]
        if target_rows.empty:
            # Use the last available date
            target_rows = ohlc.tail(1)

        row = target_rows.iloc[-1]
        idx = target_rows.index[-1]

        # Safety: idx must be within DataFrame bounds
        if idx < 0 or idx >= len(ohlc):
            logger.warning("Target row index %d out of range (len=%d)", idx, len(ohlc))
            return None

        # Get previous day
        prev_idx = idx - 1 if idx > 0 else 0
        prev_row = ohlc.iloc[prev_idx]

        snap = TechnicalSnapshot(
            date=target_date,
            spot=_safe_float(row["close"]),
            prev_close=_safe_float(prev_row["close"]),
            day_open=_safe_float(row["open"]),
            day_high=_safe_float(row["high"]),
            day_low=_safe_float(row["low"]),
            day_close=_safe_float(row["close"]),
            volume=int(_safe_float(row.get("volume", 0))),
        )
    except Exception as e:
        logger.error("Failed to build base TechnicalSnapshot: %s", e)
        return None

    close = ohlc["close"]
    high = ohlc["high"]
    low = ohlc["low"]

    # ── EMAs ──
    try:
        for period, attr in [(9, "ema_9"), (20, "ema_20"), (50, "ema_50"), (200, "ema_200")]:
            if len(close) >= period:
                ema = close.ewm(span=period, adjust=False).mean()
                val = ema.iloc[idx]
                if not pd.isna(val):
                    setattr(snap, attr, round(float(val), 2))
    except Exception as e:
        logger.debug("EMA computation failed: %s", e)

    # ── RSI ──
    try:
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            val = rsi.iloc[idx]
            if not pd.isna(val):
                snap.rsi_14 = round(float(val), 2)
    except Exception as e:
        logger.debug("RSI computation failed: %s", e)

    # ── MACD ──
    try:
        if len(close) >= 35:
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            ml = macd_line.iloc[idx]
            sl = signal_line.iloc[idx]
            hv = histogram.iloc[idx]
            if not pd.isna(ml):
                snap.macd_line = round(float(ml), 2)
            if not pd.isna(sl):
                snap.macd_signal = round(float(sl), 2)
            if not pd.isna(hv):
                snap.macd_histogram = round(float(hv), 2)
    except Exception as e:
        logger.debug("MACD computation failed: %s", e)

    # ── ADX ──
    try:
        if len(ohlc) >= 28:
            period = 14
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            hl = high - low
            hc = (high - close.shift(1)).abs()
            lc = (low - close.shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            pdi = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            mdi = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
            adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
            val = adx_series.iloc[idx]
            if not pd.isna(val):
                snap.adx = round(float(val), 2)
            pdi_val = pdi.iloc[idx]
            mdi_val = mdi.iloc[idx]
            if not pd.isna(pdi_val):
                snap.plus_di = round(float(pdi_val), 2)
            if not pd.isna(mdi_val):
                snap.minus_di = round(float(mdi_val), 2)
    except Exception as e:
        logger.debug("ADX computation failed: %s", e)

    # ── Bollinger Bands ──
    try:
        if len(close) >= 20:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_u = sma20 + 2 * std20
            bb_l = sma20 - 2 * std20
            bu = bb_u.iloc[idx]
            sm = sma20.iloc[idx]
            bl = bb_l.iloc[idx]
            if not pd.isna(bu):
                snap.bb_upper = round(float(bu), 2)
            if not pd.isna(sm):
                snap.bb_middle = round(float(sm), 2)
            if not pd.isna(bl):
                snap.bb_lower = round(float(bl), 2)
            if not pd.isna(sm) and sm > 0 and not pd.isna(bu) and not pd.isna(bl):
                snap.bb_width_pct = round(float((bu - bl) / sm * 100), 2)
    except Exception as e:
        logger.debug("Bollinger Bands computation failed: %s", e)

    # ── Supertrend ──
    try:
        if len(ohlc) >= 12:
            period_st = 10
            multiplier_st = 3.0
            hl_st = high - low
            hc_st = (high - close.shift(1)).abs()
            lc_st = (low - close.shift(1)).abs()
            tr_st = pd.concat([hl_st, hc_st, lc_st], axis=1).max(axis=1)
            atr_st = tr_st.rolling(period_st).mean()
            mid_st = (high + low) / 2
            upper_band = mid_st + multiplier_st * atr_st
            lower_band = mid_st - multiplier_st * atr_st
            direction = [True] * len(ohlc)
            for i in range(period_st + 1, len(ohlc)):
                if close.iloc[i] > upper_band.iloc[i - 1]:
                    direction[i] = True
                elif close.iloc[i] < lower_band.iloc[i - 1]:
                    direction[i] = False
                else:
                    direction[i] = direction[i - 1]
            if idx < len(direction):
                snap.supertrend_direction = "bullish" if direction[idx] else "bearish"
    except Exception as e:
        logger.debug("Supertrend computation failed: %s", e)

    # ── ATR ──
    try:
        if len(ohlc) >= 15:
            hl_atr = high - low
            hc_atr = (high - close.shift(1)).abs()
            lc_atr = (low - close.shift(1)).abs()
            tr_atr = pd.concat([hl_atr, hc_atr, lc_atr], axis=1).max(axis=1)
            atr_14 = tr_atr.rolling(14).mean()
            val = atr_14.iloc[idx]
            if not pd.isna(val):
                snap.atr_14 = round(float(val), 2)
                if snap.spot > 0:
                    snap.atr_pct = round(float(val / snap.spot * 100), 2)
    except Exception as e:
        logger.debug("ATR computation failed: %s", e)

    # ── Pivot Points (from previous day) ──
    try:
        ph = _safe_float(prev_row["high"])
        pl = _safe_float(prev_row["low"])
        pc = _safe_float(prev_row["close"])
        if ph > 0 and pl > 0 and pc > 0:
            snap.pivot = round((ph + pl + pc) / 3, 2)
            snap.r1 = round(2 * snap.pivot - pl, 2)
            snap.r2 = round(snap.pivot + (ph - pl), 2)
            snap.s1 = round(2 * snap.pivot - ph, 2)
            snap.s2 = round(snap.pivot - (ph - pl), 2)
    except Exception as e:
        logger.debug("Pivot Points computation failed: %s", e)

    # ── Gap ──
    try:
        if snap.prev_close > 0:
            gap = ((snap.day_open - snap.prev_close) / snap.prev_close) * 100
            snap.gap_pct = round(gap, 2)
            if abs(gap) < 0.2:
                snap.gap_direction = "flat"
            elif gap > 0:
                snap.gap_direction = "gap_up"
            else:
                snap.gap_direction = "gap_down"
    except Exception as e:
        logger.debug("Gap computation failed: %s", e)

    return snap


# ══════════════════════════════════════════════════════════════════════════
#  OPTIONS SNAPSHOT — from bhavcopy
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class OptionsSnapshot:
    """Options chain analytics for a specific date."""
    date: date
    has_data: bool = False
    pcr: float = 0.0
    total_call_oi: int = 0
    total_put_oi: int = 0
    max_pain: Optional[float] = None
    highest_call_oi_strike: Optional[float] = None
    highest_put_oi_strike: Optional[float] = None
    atm_iv: Optional[float] = None
    iv_skew: Optional[str] = None
    total_call_chg_oi: int = 0
    total_put_chg_oi: int = 0
    oi_buildup: Optional[str] = None  # "call_writing" / "put_writing" / etc.
    nearest_expiry: Optional[date] = None
    num_expiries: int = 0
    chain_df: Optional[pd.DataFrame] = None  # for downstream use


def compute_options_snapshot(
    chain_df: pd.DataFrame, spot: float, target_date: date,
) -> OptionsSnapshot:
    """Compute options analytics from bhavcopy data.

    The max-pain calculation uses vectorised NumPy instead of the naive
    O(strikes * rows) Python loop for much better performance on large chains.
    """
    snap = OptionsSnapshot(date=target_date)

    if chain_df is None or chain_df.empty:
        return snap

    # Guard: require option_type column
    if "option_type" not in chain_df.columns:
        logger.warning("chain_df missing 'option_type' column — skipping options snapshot")
        return snap

    # Drop rows with NaN strikes if strike column exists
    if "strike" in chain_df.columns:
        chain_df = chain_df.dropna(subset=["strike"])
        if chain_df.empty:
            return snap

    snap.has_data = True
    snap.chain_df = chain_df

    # PCR
    try:
        call_oi = chain_df[chain_df["option_type"] == "CE"]["oi"].sum() if "oi" in chain_df.columns else 0
        put_oi = chain_df[chain_df["option_type"] == "PE"]["oi"].sum() if "oi" in chain_df.columns else 0
        snap.total_call_oi = int(call_oi)
        snap.total_put_oi = int(put_oi)
        snap.pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 0.0
    except Exception as e:
        logger.debug("PCR computation failed: %s", e)

    # Change in OI
    try:
        if "chg_oi" in chain_df.columns:
            call_chg = chain_df[chain_df["option_type"] == "CE"]["chg_oi"].sum()
            put_chg = chain_df[chain_df["option_type"] == "PE"]["chg_oi"].sum()
            snap.total_call_chg_oi = int(call_chg)
            snap.total_put_chg_oi = int(put_chg)
            # Classify buildup
            if call_chg > 0 and call_chg > put_chg:
                snap.oi_buildup = "call_writing"
            elif put_chg > 0 and put_chg > call_chg:
                snap.oi_buildup = "put_writing"
            elif call_chg < 0 and put_chg > 0:
                snap.oi_buildup = "long_buildup"
            elif put_chg < 0 and call_chg > 0:
                snap.oi_buildup = "short_buildup"
    except Exception as e:
        logger.debug("OI buildup computation failed: %s", e)

    # Max Pain — vectorised O(strikes) instead of O(strikes * rows) iterrows
    try:
        if "strike" in chain_df.columns and "oi" in chain_df.columns:
            calls = chain_df[chain_df["option_type"] == "CE"]
            puts = chain_df[chain_df["option_type"] == "PE"]
            call_strikes = calls["strike"].values
            call_oi_arr = calls["oi"].values
            put_strikes = puts["strike"].values
            put_oi_arr = puts["oi"].values

            strikes = np.unique(chain_df["strike"].values)
            if len(strikes) > 0:
                min_pain = float("inf")
                max_pain_strike = None
                for s in strikes:
                    # For each candidate strike, sum the ITM pain:
                    # CE pain = max(0, s - K_call) * OI_call  (calls expire ITM when s > K)
                    # PE pain = max(0, K_put - s) * OI_put    (puts expire ITM when K > s)
                    ce_pain = np.sum(np.maximum(0, s - call_strikes) * call_oi_arr)
                    pe_pain = np.sum(np.maximum(0, put_strikes - s) * put_oi_arr)
                    pain = ce_pain + pe_pain
                    if pain < min_pain:
                        min_pain = pain
                        max_pain_strike = float(s)
                snap.max_pain = max_pain_strike
    except Exception as e:
        logger.debug("Max pain computation failed: %s", e)

    # Highest OI strikes
    try:
        if "oi" in chain_df.columns:
            calls = chain_df[chain_df["option_type"] == "CE"]
            puts = chain_df[chain_df["option_type"] == "PE"]
            if not calls.empty and calls["oi"].notna().any():
                snap.highest_call_oi_strike = float(calls.loc[calls["oi"].idxmax(), "strike"])
            if not puts.empty and puts["oi"].notna().any():
                snap.highest_put_oi_strike = float(puts.loc[puts["oi"].idxmax(), "strike"])
    except Exception as e:
        logger.debug("Highest OI strikes computation failed: %s", e)

    # Expiry info
    try:
        if "expiry" in chain_df.columns:
            expiries = sorted(chain_df["expiry"].dropna().unique())
            snap.num_expiries = len(expiries)
            if expiries:
                snap.nearest_expiry = expiries[0] if isinstance(expiries[0], date) else pd.to_datetime(expiries[0]).date()
    except Exception as e:
        logger.debug("Expiry info computation failed: %s", e)

    return snap


# ══════════════════════════════════════════════════════════════════════════
#  FULL DAY ANALYSIS — the main entry point
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class DayAnalysis:
    """Complete analysis for a single trading day."""
    date: date
    symbol: str
    display_name: str

    # Data availability
    has_ohlc: bool = False
    has_vix: bool = False
    has_options: bool = False

    # Raw data
    ohlc_df: Optional[pd.DataFrame] = None
    vix_value: float = 0.0
    vix_prev: float = 0.0
    vix_change_pct: float = 0.0

    # Analysis results
    technical: Optional[TechnicalSnapshot] = None
    options: Optional[OptionsSnapshot] = None

    # Regime classification
    day_classification: Optional[object] = None  # DayClassification
    recommendations: list = field(default_factory=list)
    risk_context: Optional[object] = None  # RiskContext

    # Metadata
    data_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def analyze_date(
    target_date: date,
    symbol: str = "NIFTY",
    lookback_days: int = 60,
) -> DayAnalysis:
    """
    Run the complete analysis pipeline for any historical date.
    
    Fetches real OHLC from yfinance, VIX from yfinance, and option chain
    from bhavcopy cache. Computes all technical indicators, options analytics,
    day classification, and strategy recommendations.
    
    This function **never raises** — it always returns a ``DayAnalysis``,
    populating as many fields as possible and recording failures in
    ``warnings``.

    Args:
        target_date: The date to analyze
        symbol: Index symbol (NIFTY, BANKNIFTY, FINNIFTY)
        lookback_days: Number of days of history to fetch for indicator computation
    
    Returns:
        DayAnalysis with all computed results (possibly partially empty on error)
    """
    cfg = INDEX_CONFIG.get(symbol, {})
    analysis = DayAnalysis(
        date=target_date,
        symbol=symbol,
        display_name=cfg.get("display_name", symbol),
    )

    ohlc: Optional[pd.DataFrame] = None

    # ── Step 1: Fetch real OHLC ──
    try:
        t0 = time.monotonic()
        ohlc = fetch_historical_ohlc(symbol, target_date, lookback_days)
        elapsed = time.monotonic() - t0
        if ohlc is not None and not ohlc.empty:
            analysis.has_ohlc = True
            analysis.ohlc_df = ohlc
            analysis.data_sources.append(
                f"yfinance ({cfg.get('yf_ticker', symbol)}) - {len(ohlc)} rows [{elapsed:.1f}s]"
            )
        else:
            analysis.warnings.append(f"No OHLC data from yfinance for {symbol} on {target_date}")
    except Exception as e:
        logger.error("Step 1 (OHLC fetch) failed: %s", e)
        analysis.warnings.append(f"OHLC fetch error: {e}")

    # ── Step 2: Fetch VIX ──
    try:
        t0 = time.monotonic()
        vix_df = fetch_historical_vix(target_date, lookback_days=10)
        elapsed = time.monotonic() - t0
        if vix_df is not None and not vix_df.empty:
            analysis.has_vix = True
            # Get VIX for target date (or nearest prior)
            target_vix = vix_df[vix_df["date"] <= target_date]
            if not target_vix.empty:
                analysis.vix_value = float(target_vix.iloc[-1]["vix"])
                if len(target_vix) >= 2:
                    analysis.vix_prev = float(target_vix.iloc[-2]["vix"])
                    if analysis.vix_prev > 0:
                        analysis.vix_change_pct = round(
                            ((analysis.vix_value - analysis.vix_prev) / analysis.vix_prev) * 100, 2
                        )
            analysis.data_sources.append(f"yfinance (India VIX) - {len(vix_df)} rows [{elapsed:.1f}s]")
        else:
            analysis.warnings.append("No VIX data available from yfinance")
    except Exception as e:
        logger.error("Step 2 (VIX fetch) failed: %s", e)
        analysis.warnings.append(f"VIX fetch error: {e}")

    # ── Step 3: Compute technical indicators ──
    try:
        if analysis.has_ohlc and ohlc is not None:
            analysis.technical = compute_technical_snapshot(ohlc, target_date)
        else:
            analysis.warnings.append("Cannot compute technicals without OHLC data")
    except Exception as e:
        logger.error("Step 3 (technicals) failed: %s", e)
        analysis.warnings.append(f"Technical computation error: {e}")

    # ── Step 4: Load option chain from bhavcopy (auto-download if missing) ──
    chain_df: Optional[pd.DataFrame] = None
    try:
        t0 = time.monotonic()
        chain_df = load_bhavcopy_for_date(target_date, symbol)

        # If not in cache, attempt auto-download from NSE
        if chain_df is None:
            try:
                from services.bhavcopy.downloader import download_bhavcopies
                from services.bhavcopy.cache import already_have

                if not already_have(target_date):
                    logger.info("Auto-downloading bhavcopy for %s...", target_date)
                    dl_result = download_bhavcopies(
                        start=target_date, end=target_date,
                    )
                    elapsed_dl = time.monotonic() - t0

                    if dl_result.total_downloaded > 0:
                        # Retry loading after successful download
                        chain_df = load_bhavcopy_for_date(target_date, symbol)
                        analysis.data_sources.append(
                            f"Bhavcopy auto-downloaded from NSE [{elapsed_dl:.1f}s]"
                        )
                        if chain_df is None:
                            # Downloaded the file but no data for this symbol
                            analysis.warnings.append(
                                f"Bhavcopy downloaded for {target_date} but no "
                                f"{symbol} options data found in the file."
                            )
                    else:
                        # Download attempted but nothing obtained
                        # Check if it's a weekend or holiday
                        wd = target_date.weekday()
                        if wd >= 5:
                            analysis.warnings.append(
                                f"{target_date} is a {'Saturday' if wd == 5 else 'Sunday'} "
                                f"— no F&O trading data available."
                            )
                        elif target_date == date.today():
                            analysis.warnings.append(
                                f"Bhavcopy for today ({target_date}) is not yet "
                                f"available — NSE publishes it after market close "
                                f"(~6 PM IST). Live options data from the Options "
                                f"Chain tab is used instead."
                            )
                        else:
                            analysis.warnings.append(
                                f"No bhavcopy available for {target_date}. "
                                f"This is likely a market holiday or the data "
                                f"is not yet published on NSE."
                            )
                        for msg in dl_result.messages:
                            logger.info("Download detail: %s", msg)
                else:
                    # File exists on disk but load returned None — wrong symbol or parse issue
                    analysis.warnings.append(
                        f"Bhavcopy file exists for {target_date} but no "
                        f"{symbol} options data could be extracted."
                    )

            except ImportError:
                logger.warning("Bhavcopy download services not available")
                analysis.warnings.append(
                    f"No bhavcopy data for {symbol} on {target_date} "
                    f"and download services are not available."
                )
            except Exception as dl_err:
                logger.error("Bhavcopy auto-download failed: %s", dl_err)
                analysis.warnings.append(
                    f"Bhavcopy auto-download failed for {target_date}: {dl_err}"
                )

        if chain_df is not None and not chain_df.empty:
            analysis.has_options = True
            spot = analysis.technical.spot if analysis.technical else 0.0
            analysis.options = compute_options_snapshot(chain_df, spot, target_date)
            elapsed = time.monotonic() - t0
            # Only add "Bhavcopy cache" source if we didn't already add auto-download source
            if not any("auto-downloaded" in s for s in analysis.data_sources):
                analysis.data_sources.append(
                    f"Bhavcopy cache - {len(chain_df)} rows [{elapsed:.1f}s]"
                )
            else:
                # Update existing source with row count
                for idx, s in enumerate(analysis.data_sources):
                    if "auto-downloaded" in s:
                        analysis.data_sources[idx] = (
                            f"Bhavcopy auto-downloaded from NSE - "
                            f"{len(chain_df)} rows [{elapsed:.1f}s]"
                        )
                        break

    except Exception as e:
        logger.error("Step 4 (options) failed: %s", e)
        analysis.warnings.append(f"Options data error: {e}")

    # ── Step 5: Classify the day ──
    try:
        from core.market_regime import (
            classify_day, recommend_strategies, RiskContext,
            GapAnalysis, CPRAnalysis,
        )

        # Build gap analysis from real data
        gap = None
        if analysis.technical and analysis.technical.gap_pct is not None:
            gap = GapAnalysis(
                gap_pct=analysis.technical.gap_pct,
                direction=analysis.technical.gap_direction or "flat",
                magnitude=(
                    "none" if abs(analysis.technical.gap_pct) < 0.2
                    else "small" if abs(analysis.technical.gap_pct) < 0.5
                    else "medium" if abs(analysis.technical.gap_pct) < 1.0
                    else "large"
                ),
            )

        # Build CPR from real previous day data
        cpr = None
        if analysis.has_ohlc and ohlc is not None and len(ohlc) >= 2:
            prev = ohlc.iloc[-2]
            spot = analysis.technical.spot if analysis.technical else float(ohlc.iloc[-1]["close"])
            cpr = CPRAnalysis.from_prev_day(
                float(prev["high"]), float(prev["low"]), float(prev["close"]), spot,
            )

        pcr_value = analysis.options.pcr if analysis.options else 0.0
        adx_value = analysis.technical.adx if analysis.technical else None

        # Compute hard-data inputs for classification
        tech = analysis.technical
        day_range_pct = None
        day_change_pct = None
        rsi_value = None
        atr_pct_val = None

        if tech:
            if tech.day_close and tech.day_close > 0:
                day_range_pct = (tech.day_high - tech.day_low) / tech.day_close * 100
            if tech.prev_close and tech.prev_close > 0:
                day_change_pct = (tech.day_close - tech.prev_close) / tech.prev_close * 100
            rsi_value = tech.rsi_14
            atr_pct_val = tech.atr_pct

        analysis.day_classification = classify_day(
            vix_value=analysis.vix_value,
            pcr_value=pcr_value,
            adx_value=adx_value,
            gap_analysis=gap,
            cpr_analysis=cpr,
            today=target_date,
            day_range_pct=day_range_pct,
            day_change_pct=day_change_pct,
            rsi_value=rsi_value,
            atr_pct=atr_pct_val,
        )

        analysis.recommendations = recommend_strategies(analysis.day_classification)
        analysis.risk_context = RiskContext(
            capital=500000,
            position_size_multiplier=analysis.day_classification.position_size_multiplier,
        )

    except Exception as e:
        logger.error("Step 5 (day classification) failed: %s", e)
        analysis.warnings.append(f"Day classification error: {e}")

    return analysis


def get_available_analysis_dates(symbol: str = "NIFTY", period: str = "6mo") -> list[date]:
    """Get list of dates that have yfinance OHLC data available."""
    cfg = INDEX_CONFIG.get(symbol)
    if not cfg:
        return []
    try:
        tk = yf.Ticker(cfg["yf_ticker"])
        df = tk.history(period=period)
        if df.empty:
            return []
        dates = pd.to_datetime(df.index).date.tolist()
        return sorted(dates)
    except Exception:
        return []
