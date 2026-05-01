"""Sentiment-vs-returns backtest.

Strategy is intentionally simple to make the research point clear:
- Compute next-day forward return per ticker.
- Median-split the prior-day sentiment per ticker.
- Long when sentiment > median, short when sentiment <= median.
- Equal-weight portfolio across tickers in the universe.

Outputs correlation, daily P&L, equity curve, and an annualized Sharpe.
NOT a production trading system — intended as a research artifact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from . import RANDOM_STATE  # noqa: F401  (kept for reproducibility note)


@dataclass(slots=True)
class BacktestResult:
    correlation: float
    sharpe: float
    cumulative_return: float
    n_days: int
    daily_pnl: pd.Series
    equity_curve: pd.Series
    aligned: pd.DataFrame  # date-indexed: sentiment, signal, fwd_ret per ticker


def fetch_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily close-to-close returns. Wide DataFrame indexed by date."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    closes = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                closes[t] = data[t]["Close"]
            else:  # single ticker
                closes[t] = data["Close"]
        except KeyError:
            continue
    px = pd.DataFrame(closes).sort_index()
    rets = px.pct_change().dropna(how="all")
    rets.index = pd.to_datetime(rets.index).tz_localize(None)
    return rets


def align(sentiment_wide: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Align sentiment with NEXT-day returns (no look-ahead).

    Returns long DataFrame with columns: date, ticker, sentiment, fwd_ret.
    """
    if sentiment_wide.empty or returns.empty:
        return pd.DataFrame(columns=["date", "ticker", "sentiment", "fwd_ret"])
    common = sorted(set(sentiment_wide.columns) & set(returns.columns))
    if not common:
        return pd.DataFrame(columns=["date", "ticker", "sentiment", "fwd_ret"])
    s = sentiment_wide[common].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    fwd = returns[common].shift(-1)  # tomorrow's return on today's row
    s_long = s.stack().rename("sentiment").reset_index()
    s_long.columns = ["date", "ticker", "sentiment"]
    f_long = fwd.stack().rename("fwd_ret").reset_index()
    f_long.columns = ["date", "ticker", "fwd_ret"]
    out = s_long.merge(f_long, on=["date", "ticker"], how="inner").dropna()
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def median_split_signal(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, sign(sentiment - median(sentiment)) → +1 long, -1 short.

    With <3 sentiment days for a ticker, fall back to sign(sentiment) so the
    research backtest still produces a defensible signal under tight data.
    """
    if aligned.empty:
        aligned = aligned.copy()
        aligned["signal"] = []
        return aligned
    out = aligned.copy()
    counts = out.groupby("ticker")["sentiment"].transform("size")
    out["med"] = out.groupby("ticker")["sentiment"].transform("median")
    median_signal = np.where(out["sentiment"] > out["med"], 1, -1)
    sign_signal = np.where(out["sentiment"] >= 0, 1, -1)
    out["signal"] = np.where(counts >= 3, median_signal, sign_signal)
    return out.drop(columns=["med"])


def run_backtest(
    sentiment_wide: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
    returns: pd.DataFrame | None = None,
) -> BacktestResult:
    if returns is None:
        returns = fetch_returns(tickers, start=start, end=end)
    aligned = align(sentiment_wide, returns)
    aligned = median_split_signal(aligned)
    if aligned.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            correlation=float("nan"),
            sharpe=float("nan"),
            cumulative_return=float("nan"),
            n_days=0,
            daily_pnl=empty,
            equity_curve=empty,
            aligned=aligned,
        )

    aligned["pnl"] = aligned["signal"] * aligned["fwd_ret"]
    daily_pnl = aligned.groupby("date")["pnl"].mean().sort_index()
    equity = (1.0 + daily_pnl).cumprod()
    cum_ret = float(equity.iloc[-1] - 1.0) if len(equity) else float("nan")

    if daily_pnl.std(ddof=0) and len(daily_pnl) > 1:
        sharpe = float(daily_pnl.mean() / daily_pnl.std(ddof=0) * np.sqrt(252))
    else:
        sharpe = float("nan")

    corr_df = aligned[["sentiment", "fwd_ret"]].dropna()
    correlation = float(corr_df.corr().iloc[0, 1]) if len(corr_df) > 1 else float("nan")

    return BacktestResult(
        correlation=correlation,
        sharpe=sharpe,
        cumulative_return=cum_ret,
        n_days=int(len(daily_pnl)),
        daily_pnl=daily_pnl,
        equity_curve=equity,
        aligned=aligned,
    )
