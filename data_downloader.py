"""
Automatic F&O bhavcopy data acquisition.

Tries multiple free sources in order until data is successfully downloaded:
  1. NSE new-format archive (nsearchives.nseindia.com) — works for 2024+
  2. NSE old-format archive (archives.nseindia.com) — works for <=mid-2024
  3. jugaad-data Python package (if installed)

All files are saved to BHAVCOPY_CACHE_DIR (~/.indian-options/bhavcopy/).
"""

import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from config import BHAVCOPY_CACHE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

# ── NSE URL templates ────────────────────────────────────────────────────────
# New format (2024-01-01 onwards, confirmed working)
_NSE_NEW_FO_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
# Old format (up to ~mid-2024)
_NSE_OLD_FO_URL = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES"
    "/{year}/{month}/fo{date}bhav.csv.zip"
)

# ── Metadata bookkeeping ────────────────────────────────────────────────────
_ACCESS_LOG = DATA_DIR / "bhavcopy_access.json"


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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _trading_days(start: date, end: date) -> list[date]:
    """Generate weekdays between start and end (inclusive).
    Holidays are not filtered — downloaders simply skip failed dates."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _new_format_filename(d: date) -> str:
    """New-format filename: BhavCopy_NSE_FO_0_0_0_20250404_F_0000.csv.zip"""
    return f"BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def _old_format_filename(d: date) -> str:
    """Old-format filename: fo04APR2025bhav.csv.zip"""
    return f"fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip"


def _already_have(d: date) -> bool:
    """Check if we already have data for this date (any known filename variant)."""
    if not BHAVCOPY_CACHE_DIR.exists():
        return False
    candidates = [
        _new_format_filename(d),
        _old_format_filename(d),
        _old_format_filename(d).replace(".zip", ""),  # bare .csv
    ]
    for name in candidates:
        if (BHAVCOPY_CACHE_DIR / name).exists():
            return True
    return False


def _get_session():
    """Get a curl_cffi session with NSE cookies seeded."""
    from nse_client import get_nse_client
    client = get_nse_client()
    client._ensure_session()
    return client._session


_DOWNLOAD_HEADERS = {
    "Referer": "https://www.nseindia.com/",
    "Accept": "*/*",
}


# ══════════════════════════════════════════════════════════════════════════════
# Source 1 — NSE new-format archives (nsearchives.nseindia.com)
# ══════════════════════════════════════════════════════════════════════════════

def _download_nse_new_format(
    start: date,
    end: date,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Download bhavcopies using the new NSE archive format.
    URL: nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
    Returns count of newly downloaded files."""
    try:
        session = _get_session()
    except (ImportError, Exception) as exc:
        logger.info("Cannot create NSE session: %s", exc)
        return 0

    BHAVCOPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    days = _trading_days(start, end)

    for d in days:
        if _already_have(d):
            continue

        url = _NSE_NEW_FO_URL.format(yyyymmdd=d.strftime("%Y%m%d"))
        try:
            resp = session.get(url, headers=_DOWNLOAD_HEADERS, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 500:
                dest = BHAVCOPY_CACHE_DIR / _new_format_filename(d)
                dest.write_bytes(resp.content)
                touch_access(dest)
                downloaded += 1
                if progress_cb:
                    progress_cb(f"NSE (new format): downloaded {d.isoformat()}")
                logger.info("Downloaded %s (new format)", dest.name)
            else:
                logger.info(
                    "NSE new-format %s -> HTTP %d (%d bytes)",
                    d.isoformat(), resp.status_code, len(resp.content),
                )
        except Exception as exc:
            logger.info("NSE new-format %s failed: %s", d.isoformat(), exc)

        # Small delay to avoid rate-limiting
        time.sleep(0.3)

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# Source 2 — NSE old-format archives (archives.nseindia.com)
# ══════════════════════════════════════════════════════════════════════════════

def _download_nse_old_format(
    start: date,
    end: date,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Download bhavcopies using the old NSE archive format.
    URL: archives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/fo{ddMMMYYYY}bhav.csv.zip
    Works for dates up to ~mid-2024. Returns count of newly downloaded files."""
    try:
        session = _get_session()
    except (ImportError, Exception) as exc:
        logger.info("Cannot create NSE session: %s", exc)
        return 0

    BHAVCOPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    days = _trading_days(start, end)

    for d in days:
        if _already_have(d):
            continue

        month_str = d.strftime("%b").upper()
        date_str = d.strftime("%d%b%Y").upper()
        url = _NSE_OLD_FO_URL.format(year=d.year, month=month_str, date=date_str)

        try:
            resp = session.get(url, headers=_DOWNLOAD_HEADERS, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 500:
                dest = BHAVCOPY_CACHE_DIR / _old_format_filename(d)
                dest.write_bytes(resp.content)
                touch_access(dest)
                downloaded += 1
                if progress_cb:
                    progress_cb(f"NSE (old format): downloaded {d.isoformat()}")
                logger.info("Downloaded %s (old format)", dest.name)
            else:
                logger.info(
                    "NSE old-format %s -> HTTP %d (%d bytes)",
                    d.isoformat(), resp.status_code, len(resp.content),
                )
        except Exception as exc:
            logger.info("NSE old-format %s failed: %s", d.isoformat(), exc)

        time.sleep(0.3)

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# Source 3 — jugaad-data package
# ══════════════════════════════════════════════════════════════════════════════

def _download_jugaad_data(
    start: date,
    end: date,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Use the jugaad-data package to download bhavcopies.
    Returns count of newly downloaded files."""
    try:
        from jugaad_data.nse import bhavcopy_fo_save  # type: ignore[import]
    except ImportError:
        logger.info("jugaad-data not installed — skipping")
        if progress_cb:
            progress_cb("jugaad-data not installed, trying next source...")
        return 0

    BHAVCOPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    days = _trading_days(start, end)

    for d in days:
        if _already_have(d):
            continue
        try:
            bhavcopy_fo_save(d, str(BHAVCOPY_CACHE_DIR))
            # jugaad-data saves with its own naming — find and mark accessed
            for f in BHAVCOPY_CACHE_DIR.iterdir():
                if d.strftime("%d%b%Y").upper() in f.name.upper():
                    touch_access(f)
            downloaded += 1
            if progress_cb:
                progress_cb(f"jugaad-data: downloaded {d.isoformat()}")
            logger.info("Downloaded %s via jugaad-data", d.isoformat())
        except Exception as exc:
            logger.info("jugaad-data %s failed: %s", d.isoformat(), exc)

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# Master downloader — tries all sources
# ══════════════════════════════════════════════════════════════════════════════

_SOURCES = [
    ("NSE Archives (new format)", _download_nse_new_format),
    ("NSE Archives (old format)", _download_nse_old_format),
    ("jugaad-data (package)", _download_jugaad_data),
]


class DownloadResult:
    """Summary of a download attempt."""

    def __init__(self) -> None:
        self.total_downloaded: int = 0
        self.source_results: dict[str, int] = {}
        self.messages: list[str] = []
        self.needed: int = 0
        self.already_had: int = 0

    @property
    def success(self) -> bool:
        return self.total_downloaded > 0

    @property
    def all_dates_covered(self) -> bool:
        return self.needed > 0 and self.total_downloaded + self.already_had >= self.needed


def download_bhavcopies(
    start: date,
    end: date,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> DownloadResult:
    """
    Try all available sources in order to download F&O bhavcopies
    for the given date range.

    Each source only attempts dates not already present on disk.
    Returns a DownloadResult summarising what happened.
    """
    result = DownloadResult()
    days = _trading_days(start, end)
    result.needed = len(days)
    result.already_had = sum(1 for d in days if _already_have(d))

    if result.already_had == result.needed:
        result.messages.append(
            f"All {result.needed} trading days already cached."
        )
        return result

    missing = result.needed - result.already_had
    result.messages.append(
        f"Need data for {missing} of {result.needed} trading days "
        f"({result.already_had} already cached)."
    )

    for source_name, source_fn in _SOURCES:
        # Re-check how many are still missing
        still_missing = sum(1 for d in days if not _already_have(d))
        if still_missing == 0:
            result.messages.append("All dates covered!")
            break

        if progress_cb:
            progress_cb(f"Trying {source_name} ({still_missing} dates remaining)...")

        try:
            count = source_fn(start, end, progress_cb=progress_cb)
        except Exception as exc:
            count = 0
            logger.error("%s failed: %s", source_name, exc)
            result.messages.append(f"{source_name}: error — {exc}")

        result.source_results[source_name] = count
        result.total_downloaded += count

        if count > 0:
            result.messages.append(f"{source_name}: downloaded {count} files.")
        else:
            result.messages.append(f"{source_name}: no new files obtained.")

    # Final tally
    final_missing = sum(1 for d in days if not _already_have(d))
    if final_missing > 0:
        result.messages.append(
            f"{final_missing} dates still missing (likely holidays or "
            f"files not yet available on NSE)."
        )

    return result


# ── Filename patterns for date extraction ────────────────────────────────────
_NEW_FMT_RE = re.compile(r"BhavCopy_NSE_FO_0_0_0_(\d{8})_F_0000", re.IGNORECASE)
_OLD_FMT_RE = re.compile(r"fo(\d{2}[A-Z]{3}\d{4})bhav", re.IGNORECASE)


def get_available_date_range() -> tuple[Optional[date], Optional[date]]:
    """Scan bhavcopy dir and return (earliest, latest) dates found."""
    if not BHAVCOPY_CACHE_DIR.exists():
        return None, None

    dates: list[date] = []
    for f in BHAVCOPY_CACHE_DIR.iterdir():
        # Try new format: BhavCopy_NSE_FO_0_0_0_20250404_F_0000
        m = _NEW_FMT_RE.search(f.name)
        if m:
            try:
                dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
            except ValueError:
                pass
            continue

        # Try old format: fo04APR2025bhav
        m = _OLD_FMT_RE.search(f.name)
        if m:
            try:
                dates.append(datetime.strptime(m.group(1), "%d%b%Y").date())
            except ValueError:
                pass

    if dates:
        return min(dates), max(dates)
    return None, None
