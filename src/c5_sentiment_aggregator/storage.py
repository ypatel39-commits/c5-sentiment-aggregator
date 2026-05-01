"""SQLite-backed cache for headlines, reddit posts, and scored sentiment.

Schema is intentionally simple: three tables keyed by source URL / id so we
can re-run the pipeline idempotently without re-downloading or re-scoring.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sentiment.db"


class Base(DeclarativeBase):
    pass


class Headline(Base):
    __tablename__ = "headlines"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(32), nullable=False)  # "yfinance" | "reddit"
    ticker = Column(String(16), nullable=False, index=True)
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    published_at = Column(DateTime, nullable=False, index=True)
    publisher = Column(String(128), nullable=True)
    __table_args__ = (UniqueConstraint("source", "url", name="uix_headline_src_url"),)


class Score(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    headline_id = Column(Integer, nullable=False, index=True, unique=True)
    label = Column(String(16), nullable=False)  # positive / neutral / negative
    score = Column(Float, nullable=False)  # signed: pos - neg in [-1, 1]
    confidence = Column(Float, nullable=False)
    model = Column(String(64), nullable=False)


def get_engine(db_path: Path | str | None = None):
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(engine) -> Iterator[Session]:
    session = Session(engine, future=True)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_headlines(engine, rows: Iterable[dict]) -> int:
    """Insert headlines, ignoring duplicates by (source, url). Returns inserted count."""
    inserted = 0
    with session_scope(engine) as s:
        for row in rows:
            existing = s.execute(
                select(Headline).where(
                    Headline.source == row["source"], Headline.url == row["url"]
                )
            ).scalar_one_or_none()
            if existing is not None:
                continue
            s.add(Headline(**row))
            inserted += 1
    return inserted


def fetch_unscored(engine) -> list[dict]:
    """Return pending headlines as plain dicts (id, ticker, title) so callers
    don't have to keep a session open.
    """
    with session_scope(engine) as s:
        scored_ids = {r[0] for r in s.execute(select(Score.headline_id)).all()}
        rows = s.execute(select(Headline)).scalars().all()
        return [
            {"id": r.id, "ticker": r.ticker, "title": r.title}
            for r in rows
            if r.id not in scored_ids
        ]


def write_scores(engine, scores: Iterable[dict]) -> int:
    inserted = 0
    with session_scope(engine) as s:
        for row in scores:
            existing = s.execute(
                select(Score).where(Score.headline_id == row["headline_id"])
            ).scalar_one_or_none()
            if existing is not None:
                continue
            s.add(Score(**row))
            inserted += 1
    return inserted


def fetch_joined(engine, ticker: str | None = None) -> list[dict]:
    """Return list of dicts joining headlines and their scores."""
    out: list[dict] = []
    with session_scope(engine) as s:
        q = select(Headline, Score).join(Score, Score.headline_id == Headline.id)
        if ticker:
            q = q.where(Headline.ticker == ticker.upper())
        for h, sc in s.execute(q).all():
            out.append(
                {
                    "ticker": h.ticker,
                    "title": h.title,
                    "url": h.url,
                    "source": h.source,
                    "published_at": h.published_at,
                    "publisher": h.publisher,
                    "label": sc.label,
                    "score": sc.score,
                    "confidence": sc.confidence,
                    "model": sc.model,
                }
            )
    return out
