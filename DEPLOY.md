# Deploy Guide — C5 Sentiment Aggregator

Deploy this Streamlit dashboard to **Streamlit Community Cloud** (free tier)
in ~5–10 minutes. The hosted app reads a pre-scored headlines parquet/CSV from
`data/`, aggregates daily sentiment per ticker, and runs a long/short backtest
against next-day returns from yfinance.

> Frame: research portfolio. **Not** production trading advice.

---

## 1. Prerequisites

- GitHub account + repo on `main` (already true).
- Streamlit Community Cloud account: <https://share.streamlit.io/>.
- A populated `data/` directory. The app calls `load_scored_frame()` which
  expects the demo data produced by `scripts/run_demo.py`. Either:
  - **Recommended:** run the demo locally and commit the resulting data
    files so the hosted app has something to display, or
  - Have `run_demo.py` invoked at deploy time (slow + brittle on free tier).

## 2. Quick deploy (5–10 min)

1. Pre-populate scored data locally:
   ```bash
   uv pip install -e .
   python scripts/run_demo.py
   git add data/ && git commit -m "data: demo scored headlines for hosted app"
   git push origin main
   ```
2. Go to <https://share.streamlit.io/> → **New app**.
3. Repository: `ypatel39-commits/c5-sentiment-aggregator`. Branch: `main`.
   Main file: `app.py`.
4. Click **Deploy**. First build takes ~3–6 minutes (numpy, pandas, scikit-learn,
   nltk, yfinance).
5. Confirm the dashboard shows daily sentiment lines, the scored feed table,
   and the backtest panel.

## 3. Files Streamlit Cloud reads

| File                       | Why                                                      |
| -------------------------- | -------------------------------------------------------- |
| `requirements.txt`         | Pip install list. Mirrors pyproject deps + `-e .`.       |
| `.streamlit/config.toml`   | Theme, server flags, telemetry off.                      |
| `app.py`                   | Entrypoint.                                              |
| `data/`                    | Pre-scored headlines for the hosted demo.                |

## 4. Environment variables

**None required** for the default scorer path. The app uses NLTK VADER /
rule-based sentiment which ships with the repo's deps.

Optional secrets you might add later (set in **App settings → Secrets**):

| Var                | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `NEWSAPI_KEY`      | If you wire NewsAPI for live headlines                   |
| `REDDIT_CLIENT_ID` | If you wire PRAW for live r/wallstreetbets ingestion     |
| `REDDIT_CLIENT_SECRET` | "                                                    |

> **Do not** enable the optional `finbert` extra (transformers + torch). It
> blows past the 1 GB free-tier RAM cap. The default lexical scorer is what
> runs in the hosted version.

## 5. Free-tier limits + expected resource usage

| Resource | Estimated usage | Risk |
| --- | --- | --- |
| RAM | 300–500 MB (pandas + sklearn + plotly) | Comfortable. |
| CPU | Spike during backtest (yfinance fetch + correlation) | Acceptable. |
| Storage | Repo `data/` ~5–30 MB depending on demo size | Fine. |
| Network | yfinance hits Yahoo Finance per backtest | Watch for rate limits. |
| Cold start | 15–30 s | Standard. |
| Sleep | Idle apps sleep after ~7 days | Wakes on visit. |

**Watchouts:**
- yfinance occasionally returns empty frames during outages — the backtest
  guards with `len(wide) < 5` but a transient empty pull produces a flat
  equity curve. Refresh.
- `data/` is **ephemeral** beyond what's in git. Do not write to it at
  runtime expecting persistence.

## 6. Troubleshooting

- **"No scored headlines yet. Run scripts/run_demo.py first."** → you didn't
  commit `data/` or it's gitignored. Generate locally, commit, push.
- **`ModuleNotFoundError: c5_sentiment_aggregator`** → confirm `-e .` is the
  last line of `requirements.txt`.
- **NLTK lookup error (`vader_lexicon` not found)** → add a one-liner to a
  startup hook or to `aggregate.py`:
  ```python
  import nltk; nltk.download("vader_lexicon", quiet=True)
  ```
- **Backtest panel says "Not enough sentiment days"** → demo data covers a
  short window. Generate more days locally and re-push.

## 7. Updating the deployed app

Push to `main`. Streamlit Cloud auto-redeploys in ~1 minute.

```bash
git add .
git commit -m "..."
git push origin main
```

---

Author: Yash Patel · Project C5 — research portfolio.
