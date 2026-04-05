"""
Robust NSE India API client using curl_cffi for TLS-fingerprint
impersonation (bypasses Akamai Bot Manager).

This module:
  - Uses curl_cffi.requests.Session with Chrome impersonation
  - Auto-fetches cookies from an NSE page before API calls
  - Implements exponential backoff with jitter on failures
  - Exposes a single .get(url) method used by the data fetcher
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Optional

from curl_cffi import requests as curl_requests

from config import NSE_BASE_URL

logger = logging.getLogger(__name__)

# Pages that reliably return 200 and set the required Akamai cookies.
# The homepage sometimes 403s; derivative/option-chain pages work better.
_COOKIE_SEED_PAGES = [
    NSE_BASE_URL + "/option-chain",
    NSE_BASE_URL + "/get-quotes/derivatives?symbol=NIFTY",
    NSE_BASE_URL,
]


class NSEClient:
    """Thread-safe (within Streamlit's model) NSE session manager."""

    _COOKIE_TTL = timedelta(minutes=4)

    def __init__(self) -> None:
        self._session: Optional[curl_requests.Session] = None
        self._cookies_set_at: Optional[datetime] = None

    # ── Session lifecycle ──────────────────────────────────────────────────

    def _new_session(self) -> curl_requests.Session:
        return curl_requests.Session(impersonate="chrome")

    def _refresh_cookies(self) -> None:
        """Visit an NSE page to obtain fresh Akamai cookies."""
        self._session = self._new_session()
        for page in _COOKIE_SEED_PAGES:
            try:
                resp = self._session.get(page, timeout=15)
                if resp.status_code == 200:
                    self._cookies_set_at = datetime.now()
                    logger.info(
                        "NSE cookies refreshed from %s (%d cookies)",
                        page, len(self._session.cookies),
                    )
                    return
                logger.debug("Seed page %s returned %d", page, resp.status_code)
            except Exception as exc:
                logger.debug("Seed page %s failed: %s", page, exc)
        # Last resort – even a 403 sometimes sets enough cookies
        self._cookies_set_at = datetime.now()
        logger.warning("Cookie refresh: none of the seed pages returned 200")

    def _ensure_session(self) -> None:
        if (
            self._session is None
            or self._cookies_set_at is None
            or datetime.now() - self._cookies_set_at > self._COOKIE_TTL
        ):
            self._refresh_cookies()

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, url: str, max_retries: int = 3) -> Optional[dict]:
        """
        GET a JSON endpoint from NSE with automatic cookie management
        and exponential backoff on failure.

        Returns the parsed JSON dict/list, or None on total failure.
        """
        self._ensure_session()

        api_headers = {
            "Referer": NSE_BASE_URL + "/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }

        for attempt in range(max_retries):
            try:
                resp = self._session.get(  # type: ignore[union-attr]
                    url, headers=api_headers, timeout=15,
                )

                if resp.status_code == 200:
                    payload = resp.json()
                    # NSE returns {} on non-trading hours for some endpoints
                    return payload

                if resp.status_code in (401, 403):
                    logger.warning(
                        "NSE %d – refreshing cookies (attempt %d)",
                        resp.status_code, attempt + 1,
                    )
                    self._refresh_cookies()
                    continue

                if resp.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning("NSE rate limit – waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                logger.warning("NSE HTTP %d for %s", resp.status_code, url)

            except Exception as exc:
                logger.warning(
                    "NSE request error (attempt %d): %s", attempt + 1, exc,
                )

            # Exponential backoff with jitter
            wait = (2 ** attempt) + random.uniform(0.3, 1.0)
            time.sleep(wait)

            # Refresh session after first failure
            if attempt == 0:
                self._refresh_cookies()

        logger.error("NSE request failed after %d attempts: %s", max_retries, url)
        return None


# Module-level singleton so Streamlit re-runs reuse the same session
_client: Optional[NSEClient] = None


def get_nse_client() -> NSEClient:
    """Return (or create) the module-level NSE client singleton."""
    global _client
    if _client is None:
        _client = NSEClient()
    return _client
