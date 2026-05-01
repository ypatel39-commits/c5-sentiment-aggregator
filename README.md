# C5 Real-Time Market Sentiment Aggregator

Research portfolio project (Yash Patel). **Not a production trading system.**

Aggregates free, public news + Reddit sentiment for a watchlist of tickers,
scores each headline with FinBERT (`yiyanghkust/finbert-tone`) — falling back
to NLTK VADER if `transformers`/`torch` are unavailable — then aligns the
daily ticker-level signal against next-day yfinance returns and produces a
simple long/short equity curve, Sharpe ratio, and signal-vs-returns
correlation.

## Why
- Practice end-to-end NLP-to-finance pipeline: ingest → score → aggregate → backtest → dashboard.
- Demonstrate handling of free data sources (yfinance `.news`, Reddit JSON, HuggingFace models) with no paid API keys.
- Honestly characterize the limits of the available data (yfinance `.news` returns only the last ~10 items per ticker, so historical backtests are inherently shallow).

## Stack
- yfinance — free news headlines + daily prices
- requests + bs4 — Reddit (`old.reddit.com/r/wallstreetbets/new.json`)
- transformers + torch + sentencepiece — FinBERT scoring (~1.5GB first load)
- nltk VADER — fallback scorer if FinBERT deps fail to install
- SQLAlchemy + SQLite — `data/sentiment.db` cache (idempotent re-runs)
- pandas, scikit-learn, matplotlib — aggregation & analytics
- streamlit + plotly — dashboard
- click — CLI for `scripts/run_demo.py`

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,finbert]"   # drop ",finbert" to use VADER fallback
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
```

## Run the demo
```bash
python scripts/run_demo.py --tickers SPY,NVDA,TSLA --backend finbert
```
Outputs:
- `data/sentiment.db` — cached headlines + scores (re-running only inserts new rows)
- `docs/sentiment-by-ticker.png` — daily mean sentiment per ticker
- `docs/equity-curve.png` — long/short equity curve

## Streamlit dashboard
```bash
streamlit run app.py
```
Ticker dropdown, sentiment time-series, scored news feed, backtest panel.

## Tests
```bash
pytest -q   # 10 tests
```

## Performance notes
- **First FinBERT call takes 30–60s** while transformers downloads the model (~1.5GB) into the HuggingFace cache. Subsequent calls are fast — the pipeline is `lru_cache`d.
- If `transformers`/`torch` installation fails (older systems), the code transparently falls back to NLTK VADER. VADER does not understand financial jargon ("beats earnings" reads as neutral) so FinBERT is preferred. The fallback is documented and exercised by the test suite.
- yfinance `.news` only returns recent items (~last 10 per ticker spanning a few days). Historical signal-vs-returns analysis is therefore inherently shallow on this free data.
- Reddit's `old.reddit.com/.../new.json` endpoint occasionally rate-limits or returns 403. The reddit module logs a warning and proceeds with news-only — the backtest does not depend on Reddit being available.

## Project layout
```
src/c5_sentiment_aggregator/
  storage.py       SQLite schema + upsert / fetch helpers
  news.py          yfinance .news ingestion
  reddit.py        r/wallstreetbets ticker-mention scraper
  score.py         FinBERT (preferred) + VADER (fallback) scorer
  aggregate.py     Daily mean sentiment per ticker
  backtest.py      Median-split long/short backtest
app.py             Streamlit dashboard
scripts/run_demo.py  End-to-end runner
tests/             pytest suite (storage, reddit, score, aggregate, backtest)
```

## Reproducibility
`RANDOM_STATE = 42` is set in `src/c5_sentiment_aggregator/__init__.py` and
used by any stochastic step (currently only the synthetic-data backtest test).

## Author
Yash Patel · Tempe, AZ · yashpatel06050@gmail.com
LinkedIn: linkedin.com/in/yash-patel-67449029b
GitHub: github.com/ypatel39-commits
