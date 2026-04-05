"""
OI Tracker – persists option chain OI snapshots to SQLite for
intraday time-series analysis across Streamlit refreshes.
"""

import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from config import OI_DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)


def _ensure_db() -> sqlite3.Connection:
    """Create DB directory and tables if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(OI_DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS oi_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            expiry TEXT NOT NULL,
            strike REAL NOT NULL,
            call_oi INTEGER DEFAULT 0,
            put_oi INTEGER DEFAULT 0,
            call_chg_oi INTEGER DEFAULT 0,
            put_chg_oi INTEGER DEFAULT 0,
            call_iv REAL DEFAULT 0,
            put_iv REAL DEFAULT 0,
            spot REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_oi_symbol_ts
        ON oi_snapshots (symbol, timestamp)
    """)
    conn.commit()
    return conn


def save_oi_snapshot(
    symbol: str,
    expiry: str,
    chain_df: pd.DataFrame,
    spot: float,
    timestamp: Optional[datetime] = None,
) -> None:
    """Save a full option chain snapshot to SQLite."""
    if chain_df.empty:
        return
    
    ts = (timestamp or datetime.now()).isoformat()
    conn = _ensure_db()
    try:
        rows = []
        for _, row in chain_df.iterrows():
            rows.append((
                ts, symbol, expiry,
                float(row.get("strike", 0)),
                int(row.get("call_oi", 0)),
                int(row.get("put_oi", 0)),
                int(row.get("call_chg_oi", 0)),
                int(row.get("put_chg_oi", 0)),
                float(row.get("call_iv", 0)),
                float(row.get("put_iv", 0)),
                spot,
            ))
        conn.executemany(
            """INSERT INTO oi_snapshots
               (timestamp, symbol, expiry, strike, call_oi, put_oi,
                call_chg_oi, put_chg_oi, call_iv, put_iv, spot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        logger.info("Saved OI snapshot: %s %s (%d strikes)", symbol, expiry, len(rows))
    except Exception as exc:
        logger.error("Failed to save OI snapshot: %s", exc)
    finally:
        conn.close()


def get_oi_timeline(
    symbol: str,
    expiry: str,
    strike: Optional[float] = None,
    trade_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Retrieve OI time-series for a symbol/expiry.
    Optionally filter to a specific strike and/or date.
    Returns DataFrame with: timestamp, strike, call_oi, put_oi, call_chg_oi, put_chg_oi, spot
    """
    conn = _ensure_db()
    try:
        query = "SELECT * FROM oi_snapshots WHERE symbol = ? AND expiry = ?"
        params: list = [symbol, expiry]

        if strike is not None:
            query += " AND strike = ?"
            params.append(strike)
        
        if trade_date is not None:
            query += " AND timestamp LIKE ?"
            params.append(f"{trade_date.isoformat()}%")

        query += " ORDER BY timestamp, strike"
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as exc:
        logger.error("Failed to read OI timeline: %s", exc)
        return pd.DataFrame()
    finally:
        conn.close()


def get_aggregate_oi_timeline(
    symbol: str,
    expiry: str,
    trade_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Get aggregated OI totals over time (sum across all strikes per timestamp).
    Returns: timestamp, total_call_oi, total_put_oi, total_call_chg, total_put_chg, pcr, spot
    """
    raw = get_oi_timeline(symbol, expiry, trade_date=trade_date)
    if raw.empty:
        return pd.DataFrame()

    agg = raw.groupby("timestamp").agg(
        total_call_oi=("call_oi", "sum"),
        total_put_oi=("put_oi", "sum"),
        total_call_chg=("call_chg_oi", "sum"),
        total_put_chg=("put_chg_oi", "sum"),
        spot=("spot", "first"),
    ).reset_index()

    agg["pcr"] = agg.apply(
        lambda r: round(r["total_put_oi"] / r["total_call_oi"], 3) if r["total_call_oi"] > 0 else 0,
        axis=1,
    )
    return agg


def get_tracked_dates(symbol: str) -> list[str]:
    """Get list of dates that have OI snapshots for a symbol."""
    conn = _ensure_db()
    try:
        rows = conn.execute(
            "SELECT DISTINCT DATE(timestamp) as dt FROM oi_snapshots WHERE symbol = ? ORDER BY dt DESC",
            (symbol,),
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def cleanup_old_data(days_to_keep: int = 30) -> None:
    """Remove OI snapshots older than N days."""
    conn = _ensure_db()
    try:
        conn.execute(
            "DELETE FROM oi_snapshots WHERE DATE(timestamp) < DATE('now', ?)",
            (f"-{days_to_keep} days",),
        )
        conn.commit()
    except Exception as exc:
        logger.error("Cleanup failed: %s", exc)
    finally:
        conn.close()
