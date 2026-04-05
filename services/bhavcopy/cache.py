"""
Shared bhavcopy cache utilities – filename patterns, access logging,
date extraction, and trading-day helpers.

Extracted from data_downloader.py and data_cleaner.py to eliminate
duplicated code.
"""

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config import BHAVCOPY_CACHE_DIR, DATA_DIR

# ── Metadata bookkeeping ────────────────────────────────────────────────────
_ACCESS_LOG = DATA_DIR / "bhavcopy_access.json"

# ── Filename patterns for date extraction ────────────────────────────────────
_NEW_FMT_RE = re.compile(r"BhavCopy_NSE_FO_0_0_0_(\d{8})_F_0000", re.IGNORECASE)
_OLD_FMT_RE = re.compile(r"fo(\d{2}[A-Z]{3}\d{4})bhav", re.IGNORECASE)


# ── Access log helpers ───────────────────────────────────────────────────────

def _load_access_log() -> dict:
    """Return {filename: last_access_iso} dict."""
    if _ACCESS_LOG.exists():
        try:
            return json.loads(_ACCESS_LOG.read_text())
        except Exception:
            return {}
    return {}


def _save_access_log(log: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ACCESS_LOG.write_text(json.dumps(log, indent=2))


def touch_access(filepath: Path) -> None:
    """Mark a bhavcopy file as 'used now'."""
    log = _load_access_log()
    log[filepath.name] = date.today().isoformat()
    _save_access_log(log)


# ── Date extraction ──────────────────────────────────────────────────────────

def extract_date_from_filename(filename: str) -> Optional[date]:
    """Extract the trading date from a bhavcopy filename (old or new format)."""
    # New format: BhavCopy_NSE_FO_0_0_0_20250404_F_0000
    m = _NEW_FMT_RE.search(filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            pass

    # Old format: fo04APR2025bhav
    m = _OLD_FMT_RE.search(filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d%b%Y").date()
        except ValueError:
            pass

    return None


# ── Trading day helpers ──────────────────────────────────────────────────────

def trading_days(start: date, end: date) -> list[date]:
    """Generate weekdays between start and end (inclusive).
    Holidays are not filtered — downloaders simply skip failed dates."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def already_have(d: date) -> bool:
    """Check if we already have data for this date (any known filename variant)."""
    if not BHAVCOPY_CACHE_DIR.exists():
        return False
    candidates = [
        new_format_filename(d),
        old_format_filename(d),
        old_format_filename(d).replace(".zip", ""),  # bare .csv
    ]
    for name in candidates:
        if (BHAVCOPY_CACHE_DIR / name).exists():
            return True
    return False


# ── Filename generators ──────────────────────────────────────────────────────

def new_format_filename(d: date) -> str:
    """New-format filename: BhavCopy_NSE_FO_0_0_0_20250404_F_0000.csv.zip"""
    return f"BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def old_format_filename(d: date) -> str:
    """Old-format filename: fo04APR2025bhav.csv.zip"""
    return f"fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
