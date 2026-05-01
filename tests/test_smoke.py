"""Package import smoke test."""

import c5_sentiment_aggregator as pkg


def test_smoke():
    assert pkg.__version__ == "0.1.0"
    assert pkg.RANDOM_STATE == 42
