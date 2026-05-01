"""Tests for aggregation and the median-split backtest using synthetic data."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from c5_sentiment_aggregator.aggregate import daily_sentiment, pivot_sentiment
from c5_sentiment_aggregator.backtest import align, median_split_signal, run_backtest


def _scored_frame() -> pd.DataFrame:
    base = datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)
    rows = []
    for d in range(8):
        for tk, base_score in (("NVDA", 0.5), ("TSLA", -0.3)):
            rows.append(
                {
                    "ticker": tk,
                    "title": f"{tk} day {d}",
                    "url": f"http://x/{tk}/{d}",
                    "source": "yfinance",
                    "published_at": base.replace(day=1 + d),
                    "publisher": "Test",
                    "label": "positive" if base_score > 0 else "negative",
                    "score": base_score + (0.1 if d % 2 == 0 else -0.1),
                    "confidence": 0.8,
                    "model": "test",
                }
            )
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def test_daily_aggregate_shape():
    df = _scored_frame()
    daily = daily_sentiment(df)
    assert set(daily.columns) == {"ticker", "date", "sentiment", "n", "confidence"}
    assert (daily["n"] == 1).all()
    wide = pivot_sentiment(daily)
    assert set(wide.columns) == {"NVDA", "TSLA"}
    assert len(wide) == 8


def test_backtest_with_injected_returns():
    df = _scored_frame()
    wide = pivot_sentiment(daily_sentiment(df))
    rng = np.random.default_rng(42)
    # Construct returns that align positively with sentiment for NVDA.
    dates = pd.date_range(wide.index.min(), wide.index.max() + pd.Timedelta(days=1))
    rets = pd.DataFrame(
        {
            "NVDA": rng.normal(0, 0.005, len(dates)),
            "TSLA": rng.normal(0, 0.005, len(dates)),
        },
        index=dates,
    )
    # Force a strong positive correlation: tomorrow's NVDA move follows today's NVDA sentiment sign.
    for d, row in wide.iterrows():
        nxt = d + pd.Timedelta(days=1)
        if nxt in rets.index:
            rets.loc[nxt, "NVDA"] = 0.01 if row["NVDA"] > wide["NVDA"].median() else -0.01
            rets.loc[nxt, "TSLA"] = 0.01 if row["TSLA"] > wide["TSLA"].median() else -0.01

    aligned = align(wide, rets)
    assert not aligned.empty
    signed = median_split_signal(aligned)
    assert set(signed["signal"].unique()) <= {-1, 1}

    res = run_backtest(wide, ["NVDA", "TSLA"], "2024-01-01", "2024-01-15", returns=rets)
    assert res.n_days > 0
    assert res.cumulative_return > 0  # constructed to be profitable
    assert not res.equity_curve.empty
