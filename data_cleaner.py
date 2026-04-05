"""
Cleanup of bhavcopy data files.

Provides:
  - cleanup_date_range(): delete cached files for a specific date range
  - get_cache_stats(): cache directory metrics
  - mark_files_accessed(): track which files are in use
"""

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config import BHAVCOPY_CACHE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

_ACCESS_LOG = DATA_DIR / "bhavcopy_access.json"

# Regex patterns to extract dates from bhavcopy filenames
_NEW_FMT_RE = re.compile(r"BhavCopy_NSE_FO_0_0_0_(\d{8})_F_0000", re.IGNORECASE)
_OLD_FMT_RE = re.compile(r"fo(\d{2}[A-Z]{3}\d{4})bhav", re.IGNORECASE)


def _load_access_log() -> dict:
    if _ACCESS_LOG.exists():
        try:
            return json.loads(_ACCESS_LOG.read_text())
        except Exception:
            return {}
    return {}


def _save_access_log(log: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ACCESS_LOG.write_text(json.dumps(log, indent=2))


def mark_files_accessed(filepaths: list[Path]) -> None:
    """Mark a list of files as 'used today'. Call this after loading
    bhavcopy files for backtesting."""
    log = _load_access_log()
    today = date.today().isoformat()
    for fp in filepaths:
        log[fp.name] = today
    _save_access_log(log)


def _extract_date_from_filename(filename: str) -> Optional[date]:
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


def cleanup_date_range(
    start: date,
    end: date,
) -> dict:
    """
    Delete bhavcopy files whose trading date falls within [start, end].

    Returns a summary dict:
        {
            "scanned": int,
            "deleted": [list of filenames],
            "kept": int,
            "errors": [list of error strings],
        }
    """
    summary = {
        "scanned": 0,
        "deleted": [],
        "kept": 0,
        "errors": [],
    }

    if not BHAVCOPY_CACHE_DIR.exists():
        return summary

    access_log = _load_access_log()
    log_changed = False

    for fp in sorted(BHAVCOPY_CACHE_DIR.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix not in (".csv", ".zip"):
            continue

        summary["scanned"] += 1

        file_date = _extract_date_from_filename(fp.name)
        if file_date is None:
            summary["kept"] += 1
            continue

        if start <= file_date <= end:
            try:
                fp.unlink()
                summary["deleted"].append(fp.name)
                if fp.name in access_log:
                    del access_log[fp.name]
                    log_changed = True
                logger.info("Deleted: %s (date: %s)", fp.name, file_date)
            except Exception as exc:
                summary["errors"].append(f"{fp.name}: {exc}")
                logger.error("Failed to delete %s: %s", fp.name, exc)
        else:
            summary["kept"] += 1

    if log_changed:
        _save_access_log(access_log)

    return summary


def get_cache_stats() -> dict:
    """Return stats about the bhavcopy cache directory."""
    stats = {
        "total_files": 0,
        "total_size_mb": 0.0,
        "oldest_file": None,
        "newest_file": None,
    }

    if not BHAVCOPY_CACHE_DIR.exists():
        return stats

    dates = []
    for fp in BHAVCOPY_CACHE_DIR.iterdir():
        if not fp.is_file() or fp.suffix not in (".csv", ".zip"):
            continue
        stats["total_files"] += 1
        try:
            stats["total_size_mb"] += fp.stat().st_size / (1024 * 1024)
        except Exception:
            pass

        d = _extract_date_from_filename(fp.name)
        if d:
            dates.append(d)

    if dates:
        stats["oldest_file"] = str(min(dates))
        stats["newest_file"] = str(max(dates))

    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    return stats
