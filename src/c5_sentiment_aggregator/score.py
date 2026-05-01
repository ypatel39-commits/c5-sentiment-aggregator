"""Sentiment scoring with FinBERT (preferred) or NLTK VADER (fallback).

The signed score in [-1, 1] is computed as P(positive) - P(negative) for FinBERT
and as the VADER 'compound' for the fallback. This makes downstream aggregation
identical regardless of backend.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

from .storage import fetch_unscored, get_engine, write_scores

log = logging.getLogger(__name__)

FINBERT_MODEL = "yiyanghkust/finbert-tone"


@lru_cache(maxsize=1)
def _load_finbert():
    """Load FinBERT pipeline. Cached so subsequent calls are fast.

    The yiyanghkust/finbert-tone repo only ships a vocab.txt (no fast
    tokenizer files), so AutoTokenizer fails. We explicitly use the slow
    BertTokenizer, which is what FinBERT was trained with.
    """
    from transformers import (  # type: ignore[import-not-found]
        BertForSequenceClassification,
        BertTokenizer,
        pipeline,
    )

    tok = BertTokenizer.from_pretrained(FINBERT_MODEL)
    mdl = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None)


@lru_cache(maxsize=1)
def _load_vader():
    import nltk  # type: ignore[import-not-found]

    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore[import-not-found]

        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore[import-not-found]

        return SentimentIntensityAnalyzer()


def _backend() -> str:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return "finbert"
    except Exception:
        return "vader"


def _score_finbert(texts: list[str]) -> list[dict]:
    pipe = _load_finbert()
    raw = pipe(texts, truncation=True)
    out: list[dict] = []
    for entry in raw:
        # entry is a list of {label, score} for each class
        probs = {d["label"].lower(): float(d["score"]) for d in entry}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        signed = pos - neg
        label = max(probs, key=probs.get)
        out.append(
            {
                "label": label,
                "score": signed,
                "confidence": float(max(probs.values())),
                "model": FINBERT_MODEL,
            }
        )
    return out


def _score_vader(texts: list[str]) -> list[dict]:
    sia = _load_vader()
    out: list[dict] = []
    for t in texts:
        s = sia.polarity_scores(t or "")
        compound = float(s["compound"])
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        out.append(
            {
                "label": label,
                "score": compound,
                "confidence": float(max(s["pos"], s["neu"], s["neg"])),
                "model": "vader",
            }
        )
    return out


def score_texts(texts: list[str], backend: str | None = None) -> list[dict]:
    """Score a list of strings. Auto-detect backend if not specified."""
    if not texts:
        return []
    backend = backend or _backend()
    if backend == "finbert":
        try:
            return _score_finbert(texts)
        except Exception as e:  # pragma: no cover - heavy import path
            log.warning("FinBERT failed (%s) — falling back to VADER", e)
            return _score_vader(texts)
    return _score_vader(texts)


def score_pending(db_path=None, backend: str | None = None) -> int:
    """Score every headline in the DB that lacks a sentiment score."""
    engine = get_engine(db_path)
    pending = fetch_unscored(engine)
    if not pending:
        return 0
    texts = [h["title"] for h in pending]
    scored = score_texts(texts, backend=backend)
    rows = [
        {
            "headline_id": h["id"],
            "label": s["label"],
            "score": s["score"],
            "confidence": s["confidence"],
            "model": s["model"],
        }
        for h, s in zip(pending, scored)
    ]
    return write_scores(engine, rows)
