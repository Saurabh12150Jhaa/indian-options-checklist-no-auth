"""
News fetcher – aggregates Indian market / derivatives news from free RSS feeds.

Sources:
  - Economic Times Markets (primary)
  - LiveMint Markets
  - Business Standard Markets
  - Google News (index-specific options / F&O queries)
"""

import logging
import re
from datetime import datetime
from typing import Optional

import feedparser
import pandas as pd

from config import IST

logger = logging.getLogger(__name__)

# ── Feed definitions ──────────────────────────────────────────────────────

_MARKET_FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "LiveMint": "https://www.livemint.com/rss/markets",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
}

_GOOGLE_NEWS_BASE = (
    "https://news.google.com/rss/search?hl=en-IN&gl=IN&ceid=IN:en&q="
)

_INDEX_QUERIES = {
    "NIFTY": "NIFTY+options+India+derivatives",
    "BANKNIFTY": "BANKNIFTY+options+Bank+Nifty",
    "FINNIFTY": "FINNIFTY+OR+%22Fin+Nifty%22+options+India",
}

# ── Category classification ───────────────────────────────────────────────

_CATEGORY_PATTERNS = {
    "Options / F&O": re.compile(
        r"option|f\s*&\s*o|derivative|expir|straddle|strangle|iron\s*condor"
        r"|call|put|strike|premium|open\s*interest|PCR|max\s*pain|implied\s*vol",
        re.IGNORECASE,
    ),
    "Market Outlook": re.compile(
        r"outlook|forecast|predict|expect|target|support|resistance|bullish|bearish"
        r"|rally|crash|correction|rebound|sentiment|market\s*today|next\s*week",
        re.IGNORECASE,
    ),
    "Macro / Policy": re.compile(
        r"RBI|repo\s*rate|monetary\s*policy|inflation|GDP|fiscal|government|budget"
        r"|tariff|crude|dollar|rupee|FII|FPI|DII|trade\s*war|geopolitical",
        re.IGNORECASE,
    ),
    "Sector / Stock": re.compile(
        r"IPO|stock\s*pick|buy|sell|hold|recommend|sector|earnings|quarterly"
        r"|result|Q[1-4]|revenue|profit|bank\s*nifty|IT\s*sector|pharma|auto",
        re.IGNORECASE,
    ),
    "Technical": re.compile(
        r"technical|chart|EMA|RSI|MACD|pivot|fibonacci|breakout|breakdown"
        r"|trend|pattern|candle|reversal|moving\s*average",
        re.IGNORECASE,
    ),
}


def _classify(title: str, summary: str) -> str:
    text = f"{title} {summary}"
    for category, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(text):
            return category
    return "General"


# ── Impact scoring ────────────────────────────────────────────────────────

_POSITIVE_RE = re.compile(
    r"rally|bullish|buy|upgrade|positive|gain|rebound|rise|boost|strong|surge|support",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"crash|bearish|sell|downgrade|negative|loss|correction|fall|drop|weak|plunge|resistance|risk",
    re.IGNORECASE,
)


def _sentiment(title: str, summary: str) -> str:
    text = f"{title} {summary}"
    pos = len(_POSITIVE_RE.findall(text))
    neg = len(_NEGATIVE_RE.findall(text))
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


# ── Public API ────────────────────────────────────────────────────────────


def fetch_news(
    index_name: str = "NIFTY",
    max_items: int = 50,
) -> tuple[pd.DataFrame, datetime]:
    """
    Fetch and classify market news relevant to *index_name*.

    Returns a DataFrame with columns:
        Source, Category, Title, Summary, Sentiment, Published, Link
    """
    all_items: list[dict] = []

    # 1. General market feeds
    for source_name, url in _MARKET_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                all_items.append(_parse_entry(entry, source_name))
        except Exception as exc:
            logger.warning("RSS %s failed: %s", source_name, exc)

    # 2. Google News – index-specific
    query = _INDEX_QUERIES.get(index_name, _INDEX_QUERIES["NIFTY"])
    try:
        feed = feedparser.parse(_GOOGLE_NEWS_BASE + query)
        for entry in feed.entries[:30]:
            all_items.append(_parse_entry(entry, f"Google ({index_name})"))
    except Exception as exc:
        logger.warning("Google News RSS failed: %s", exc)

    # 3. Google News – broad F&O
    try:
        feed = feedparser.parse(
            _GOOGLE_NEWS_BASE + "India+derivatives+F%26O+market+Nifty"
        )
        for entry in feed.entries[:20]:
            all_items.append(_parse_entry(entry, "Google (F&O)"))
    except Exception as exc:
        logger.warning("Google F&O News RSS failed: %s", exc)

    if not all_items:
        return pd.DataFrame(), datetime.now(IST)

    df = pd.DataFrame(all_items)
    # De-duplicate by title (different sources may carry the same story)
    df.drop_duplicates(subset="Title", keep="first", inplace=True)
    df.sort_values("Published", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.head(max_items), datetime.now(IST)


def _parse_entry(entry: dict, source: str) -> dict:
    title = entry.get("title", "")
    summary = _clean_html(entry.get("summary", entry.get("description", "")))
    published = entry.get("published", "")
    link = entry.get("link", "")
    # Google News embeds the actual source in a <source> element
    if hasattr(entry, "source") and isinstance(entry.source, dict):
        source = entry.source.get("title", source)
    elif "source" in entry and isinstance(entry["source"], dict):
        source = entry["source"].get("title", source)

    return {
        "Source": source,
        "Category": _classify(title, summary),
        "Title": title,
        "Summary": summary[:200] if summary else "",
        "Sentiment": _sentiment(title, summary),
        "Published": published,
        "Link": link,
    }


def _clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text).strip()
