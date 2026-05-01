"""End-to-end demo: ingest → score → aggregate → backtest → save plots.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --tickers SPY,NVDA,TSLA --backend auto
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from c5_sentiment_aggregator.aggregate import daily_sentiment, load_scored_frame, pivot_sentiment  # noqa: E402
from c5_sentiment_aggregator.backtest import run_backtest  # noqa: E402
from c5_sentiment_aggregator.news import ingest_news  # noqa: E402
from c5_sentiment_aggregator.reddit import ingest_reddit  # noqa: E402
from c5_sentiment_aggregator.score import score_pending  # noqa: E402

DEFAULT_TICKERS = "SPY,NVDA,TSLA"
DOCS = ROOT / "docs"


def _save_sentiment_plot(daily, path: Path) -> None:
    if daily.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for tk, sub in daily.groupby("ticker"):
        ax.plot(sub["date"], sub["sentiment"], marker="o", label=tk)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_title("Daily mean sentiment by ticker")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Mean sentiment (signed)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_equity_plot(equity, path: Path) -> None:
    if equity.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_title("Backtest equity curve (start = $1.00)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@click.command()
@click.option("--tickers", default=DEFAULT_TICKERS, help="Comma-separated ticker list")
@click.option("--backend", default=None, type=click.Choice(["finbert", "vader"]))
@click.option("--skip-reddit/--with-reddit", default=False)
@click.option("--start", default=None, help="Backtest start (YYYY-MM-DD)")
@click.option("--end", default=None, help="Backtest end (YYYY-MM-DD)")
def main(tickers: str, backend: str | None, skip_reddit: bool, start: str | None, end: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    DOCS.mkdir(parents=True, exist_ok=True)
    ts = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    print(f"[1/5] Ingest news for: {ts}")
    news_counts = ingest_news(ts)
    print(f"      inserted: {news_counts}")

    if not skip_reddit:
        print("[2/5] Ingest Reddit (r/wallstreetbets) — falls back gracefully on failure")
        reddit_count = ingest_reddit(ts)
        print(f"      inserted: {reddit_count}")
    else:
        print("[2/5] Reddit ingestion skipped")

    print("[3/5] Score pending headlines")
    n_scored = score_pending(backend=backend)
    print(f"      scored: {n_scored}")

    print("[4/5] Aggregate to daily ticker-level signal")
    df = load_scored_frame()
    daily = daily_sentiment(df)
    wide = pivot_sentiment(daily)
    sentiment_path = DOCS / "sentiment-by-ticker.png"
    _save_sentiment_plot(daily, sentiment_path)
    print(f"      rows: {len(daily)} | saved {sentiment_path}")

    print("[5/5] Backtest sentiment vs next-day returns")
    if wide.empty or len(wide) < 2:
        print("      not enough sentiment history to backtest")
        return
    bt_start = start or (wide.index.min() - timedelta(days=3)).strftime("%Y-%m-%d")
    bt_end = end or (wide.index.max() + timedelta(days=3)).strftime("%Y-%m-%d")
    if not start:
        bt_start = max(bt_start, (date.today() - timedelta(days=365)).strftime("%Y-%m-%d"))
    res = run_backtest(wide, list(wide.columns), start=bt_start, end=bt_end)
    eq_path = DOCS / "equity-curve.png"
    _save_equity_plot(res.equity_curve, eq_path)
    print(
        f"      correlation={res.correlation:+.4f} | sharpe={res.sharpe:+.3f} | "
        f"cum_ret={res.cumulative_return:+.4f} | n_days={res.n_days}"
    )
    print(f"      saved {eq_path}")


if __name__ == "__main__":
    main()
