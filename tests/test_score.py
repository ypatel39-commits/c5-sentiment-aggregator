"""Sentiment scoring tests — uses VADER backend (no heavy download)."""

from c5_sentiment_aggregator.score import score_texts


def test_vader_scores_directionality():
    # NOTE: VADER lacks financial domain vocab (the reason FinBERT exists);
    # we test directionality with plain emotional language.
    out = score_texts(
        [
            "amazing fantastic wonderful great success",
            "terrible awful disaster horrible failure",
            "the meeting is scheduled for tomorrow",
        ],
        backend="vader",
    )
    assert len(out) == 3
    assert out[0]["score"] > 0
    assert out[1]["score"] < 0
    for entry in out:
        assert -1.0 <= entry["score"] <= 1.0
        assert entry["label"] in {"positive", "neutral", "negative"}
        assert entry["model"] == "vader"


def test_empty_input():
    assert score_texts([], backend="vader") == []
