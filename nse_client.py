"""
Robust NSE India API client with session/cookie management.

NSE blocks automated requests aggressively, so this module:
  - Maintains a requests.Session with realistic browser headers
  - Auto-fetches cookies from the NSE homepage before API calls
  - Implements exponential backoff with jitter on failures
  - Exposes a single .get(url) method used by the data fetcher
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import NSE_BASE_URL

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) "
        "Gecko/20100101 Firefox/126.0"
    ),
]

_BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class NSEClient:
    """Thread-safe (within Streamlit's model) NSE session manager."""

    # How long a cookie set is considered valid before we refresh
    _COOKIE_TTL = timedelta(minutes=4)

    def __init__(self) -> None:
        self._session: Optional[requests.Session] = None
        self._cookies_set_at: Optional[datetime] = None

    # ── Session lifecycle ──────────────────────────────────────────────────

    def _new_session(self) -> requests.Session:
        sess = requests.Session()
        ua = random.choice(_USER_AGENTS)
        sess.headers.update({**_BASE_HEADERS, "User-Agent": ua})
        return sess

    def _refresh_cookies(self) -> None:
        """Hit the NSE homepage to obtain fresh cookies."""
        self._session = self._new_session()
        try:
            resp = self._session.get(NSE_BASE_URL, timeout=10)
            resp.raise_for_status()
            self._cookies_set_at = datetime.now()
            logger.info("NSE cookies refreshed (%d cookies)", len(self._session.cookies))
        except requests.RequestException as exc:
            logger.warning("Cookie refresh failed: %s", exc)
            self._cookies_set_at = None

    def _ensure_session(self) -> None:
        """Lazily create or refresh the session when cookies are stale."""
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

        Returns the parsed JSON dict, or None on total failure.
        """
        self._ensure_session()

        api_headers = {
            "Referer": NSE_BASE_URL + "/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }

        for attempt in range(max_retries):
            try:
                resp = self._session.get(url, headers=api_headers, timeout=15)  # type: ignore[union-attr]

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (401, 403):
                    logger.warning("NSE returned %d – refreshing cookies (attempt %d)", resp.status_code, attempt + 1)
                    self._refresh_cookies()
                    continue

                if resp.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning("NSE rate limit hit – waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                logger.warning("NSE returned HTTP %d for %s", resp.status_code, url)

            except requests.RequestException as exc:
                logger.warning("NSE request error (attempt %d): %s", attempt + 1, exc)

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
