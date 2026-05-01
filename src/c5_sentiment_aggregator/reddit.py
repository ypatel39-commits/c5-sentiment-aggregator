"""Reddit scraper for r/wallstreetbets ticker mentions.

Uses the public .json endpoint on old.reddit.com (no auth). If the request
fails we log a warning and return an empty list — the rest of the pipeline
proceeds with news only, per project spec.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Iterable

import requests

from .storage import get_engine, upsert_headlines

log = logging.getLogger(__name__)

USER_AGENT = "c5-sentiment-aggregator/0.1 (research portfolio; +https://github.com/ypatel39-commits)"
WSB_URL = "https://old.reddit.com/r/wallstreetbets/new.json?limit=100"
TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")


def _ticker_set(tickers: Iterable[str]) -> set[str]:
    return {t.upper() for t in tickers}


def _fetch_listing(url: str = WSB_URL, timeout: int = 10) -> list[dict]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        return [c.get("data", {}) for c in payload.get("data", {}).get("children", [])]
    except Exception as e:
        log.warning("reddit scrape failed (%s) — falling back to news only", e)
        return []


def extract_ticker_mentions(title: str, watchlist: set[str]) -> list[str]:
    """Return tickers from watchlist mentioned in title via $TICKER pattern."""
    found = {m.group(1) for m in TICKER_RE.finditer(title or "")}
    return sorted(found & watchlist)


def fetch_wsb_mentions(tickers: Iterable[str]) -> list[dict]:
    """Scrape WSB and emit one row per (post, mentioned ticker) match."""
    watch = _ticker_set(tickers)
    posts = _fetch_listing()
    rows: list[dict] = []
    for p in posts:
        title = p.get("title") or ""
        url = p.get("url") or p.get("permalink") or ""
        if url.startswith("/"):
            url = f"https://reddit.com{url}"
        ts = p.get("created_utc")
        published = (
            datetime.fromtimestamp(int(ts), tz=timezone.utc)
            if isinstance(ts, (int, float))
            else datetime.now(tz=timezone.utc)
        )
        for tk in extract_ticker_mentions(title, watch):
            rows.append(
                {
                    "source": "reddit",
                    "ticker": tk,
                    "title": title[:500],
                    "url": str(url),
                    "published_at": published,
                    "publisher": "r/wallstreetbets",
                }
            )
    log.info("reddit produced %d ticker-mention rows", len(rows))
    return rows


def ingest_reddit(tickers: Iterable[str], db_path=None) -> int:
    engine = get_engine(db_path)
    rows = fetch_wsb_mentions(tickers)
    return upsert_headlines(engine, rows) if rows else 0
