"""News headline fetcher using yfinance .news (free, no API key)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import yfinance as yf

from .storage import get_engine, upsert_headlines

log = logging.getLogger(__name__)


def _coerce_ts(item: dict) -> datetime:
    """yfinance news shape changed across versions. Try multiple keys."""
    for key in ("providerPublishTime", "pubDate", "published"):
        v = item.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return datetime.fromtimestamp(int(v), tz=timezone.utc)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
    content = item.get("content") or {}
    for key in ("pubDate", "displayTime"):
        v = content.get(key)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
    return datetime.now(tz=timezone.utc)


def _normalize(item: dict, ticker: str) -> dict | None:
    """Flatten a yfinance news item to our schema. Returns None if unusable."""
    title = item.get("title")
    url = item.get("link") or item.get("url")
    publisher = item.get("publisher")
    if not title or not url:
        content = item.get("content") or {}
        title = title or content.get("title")
        url = url or (content.get("canonicalUrl") or {}).get("url") or content.get("clickThroughUrl", {}).get("url")
        publisher = publisher or (content.get("provider") or {}).get("displayName")
    if not title or not url:
        return None
    return {
        "source": "yfinance",
        "ticker": ticker.upper(),
        "title": str(title)[:500],
        "url": str(url),
        "published_at": _coerce_ts(item),
        "publisher": str(publisher)[:128] if publisher else None,
    }


def fetch_news(ticker: str) -> list[dict]:
    """Pull recent news for a ticker. Returns normalized rows (may be empty)."""
    try:
        tk = yf.Ticker(ticker)
        items = tk.news or []
    except Exception as e:  # pragma: no cover - network shape errors
        log.warning("yfinance news failed for %s: %s", ticker, e)
        return []
    rows: list[dict] = []
    for it in items:
        norm = _normalize(it, ticker)
        if norm is not None:
            rows.append(norm)
    log.info("fetched %d news items for %s", len(rows), ticker)
    return rows


def ingest_news(tickers: Iterable[str], db_path=None) -> dict[str, int]:
    """Fetch and persist news for each ticker. Returns inserted counts."""
    engine = get_engine(db_path)
    counts: dict[str, int] = {}
    for t in tickers:
        rows = fetch_news(t)
        counts[t.upper()] = upsert_headlines(engine, rows) if rows else 0
    return counts
