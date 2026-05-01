"""Tests for SQLite storage layer."""
from datetime import datetime, timezone

from c5_sentiment_aggregator.storage import (
    fetch_joined,
    fetch_unscored,
    get_engine,
    upsert_headlines,
    write_scores,
)


def _hd(url: str, ticker: str = "NVDA") -> dict:
    return {
        "source": "yfinance",
        "ticker": ticker,
        "title": f"Headline {url}",
        "url": url,
        "published_at": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        "publisher": "Test",
    }


def test_upsert_dedupes_by_url(tmp_path):
    engine = get_engine(tmp_path / "t.db")
    rows = [_hd("https://x/1"), _hd("https://x/2"), _hd("https://x/1")]
    inserted = upsert_headlines(engine, rows)
    assert inserted == 2
    again = upsert_headlines(engine, [_hd("https://x/1")])
    assert again == 0


def test_score_roundtrip(tmp_path):
    engine = get_engine(tmp_path / "t.db")
    upsert_headlines(engine, [_hd("https://x/1"), _hd("https://x/2", ticker="TSLA")])
    pending = fetch_unscored(engine)
    assert len(pending) == 2
    rows = [
        {"headline_id": pending[0]["id"], "label": "positive", "score": 0.8, "confidence": 0.9, "model": "test"},
        {"headline_id": pending[1]["id"], "label": "negative", "score": -0.5, "confidence": 0.7, "model": "test"},
    ]
    written = write_scores(engine, rows)
    assert written == 2
    joined = fetch_joined(engine)
    assert len(joined) == 2
    assert {r["ticker"] for r in joined} == {"NVDA", "TSLA"}
