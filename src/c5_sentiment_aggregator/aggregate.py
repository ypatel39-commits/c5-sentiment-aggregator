"""Daily ticker-level sentiment aggregation."""
from __future__ import annotations

import pandas as pd

from .storage import fetch_joined, get_engine


def load_scored_frame(db_path=None) -> pd.DataFrame:
    engine = get_engine(db_path)
    rows = fetch_joined(engine)
    if not rows:
        return pd.DataFrame(
            columns=["ticker", "title", "url", "source", "published_at", "score", "label", "confidence"]
        )
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, per UTC date: mean signed score, count, mean confidence.

    Returns a long DataFrame with columns: ticker, date, sentiment, n, confidence.
    """
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "sentiment", "n", "confidence"])
    out = (
        df.assign(date=lambda d: d["published_at"].dt.tz_convert("UTC").dt.normalize())
        .groupby(["ticker", "date"], as_index=False)
        .agg(sentiment=("score", "mean"), n=("score", "size"), confidence=("confidence", "mean"))
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    out["date"] = out["date"].dt.tz_localize(None)
    return out


def pivot_sentiment(daily: pd.DataFrame) -> pd.DataFrame:
    """Wide pivot: index=date, columns=ticker, values=sentiment."""
    if daily.empty:
        return pd.DataFrame()
    return daily.pivot(index="date", columns="ticker", values="sentiment").sort_index()
