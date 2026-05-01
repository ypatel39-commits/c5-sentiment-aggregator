"""Tests for reddit ticker extraction."""

from c5_sentiment_aggregator.reddit import extract_ticker_mentions


def test_extracts_dollar_tickers():
    title = "YOLO into $NVDA and $TSLA before earnings"
    found = extract_ticker_mentions(title, watchlist={"NVDA", "TSLA", "SPY"})
    assert found == ["NVDA", "TSLA"]


def test_filters_to_watchlist():
    title = "Loaded up on $XYZ — to the moon"
    assert extract_ticker_mentions(title, watchlist={"NVDA"}) == []


def test_handles_empty_and_none():
    assert extract_ticker_mentions("", {"NVDA"}) == []
    assert extract_ticker_mentions(None, {"NVDA"}) == []  # type: ignore[arg-type]
