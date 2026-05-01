"""Streamlit dashboard for the C5 Sentiment Aggregator.

Run:
    streamlit run app.py

Frame: research portfolio. NOT a production trading system.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from c5_sentiment_aggregator.aggregate import daily_sentiment, load_scored_frame, pivot_sentiment
from c5_sentiment_aggregator.backtest import run_backtest

st.set_page_config(page_title="C5 Sentiment Aggregator", layout="wide")
st.title("C5 Real-Time Market Sentiment Aggregator")
st.caption("Research portfolio (Yash Patel). Not production trading advice.")


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return load_scored_frame()


df = _load()
if df.empty:
    st.warning("No scored headlines yet. Run `python scripts/run_demo.py` first.")
    st.stop()

tickers = sorted(df["ticker"].unique().tolist())
selected = st.sidebar.multiselect("Tickers", tickers, default=tickers[: min(3, len(tickers))])
view = df[df["ticker"].isin(selected)] if selected else df

st.subheader("Daily mean sentiment")
daily = daily_sentiment(view)
if not daily.empty:
    fig = px.line(daily, x="date", y="sentiment", color="ticker", markers=True)
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No daily aggregates for current selection.")

st.subheader("News & Reddit feed (scored)")
feed = view.sort_values("published_at", ascending=False).head(50)[
    ["published_at", "ticker", "source", "label", "score", "title", "publisher", "url"]
]
st.dataframe(feed, use_container_width=True, hide_index=True)

st.subheader("Backtest: long/short on sentiment vs next-day returns")
if selected:
    wide = pivot_sentiment(daily)
    if wide.empty or len(wide) < 5:
        st.info("Not enough sentiment days to backtest (need ~5+).")
    else:
        start = (wide.index.min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end = (wide.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        with st.spinner("Pulling prices and running backtest..."):
            res = run_backtest(wide, list(wide.columns), start=start, end=end)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment-fwd-ret correlation", f"{res.correlation:+.3f}")
        c2.metric("Sharpe (annualized)", f"{res.sharpe:+.2f}")
        c3.metric("Cumulative return", f"{res.cumulative_return:+.2%}")
        if not res.equity_curve.empty:
            eq = res.equity_curve.rename("equity").reset_index()
            fig2 = px.line(eq, x="date", y="equity", title="Equity curve (start = $1.00)")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select at least one ticker to run the backtest.")
