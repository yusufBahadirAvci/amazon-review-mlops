"""Tests for Ceylin's drift monitoring: monitoring/drift_report.py.

A fake model replaces the MLflow model, so no server is needed.
"""

import pandas as pd

from monitoring.drift_report import add_features, drifted_share, metric_label


class FakeModel:
    """Stands in for the MLflow model: always predicts 'positive'."""

    def predict(self, df: pd.DataFrame) -> list[str]:
        return ["positive"] * len(df)


def test_add_features_cleans_rows_and_adds_columns():
    raw = pd.DataFrame(
        {"text": ["Good product", "   ", None, "Bad one here"], "rating": [5, 3, 4, 1]}
    )
    features = add_features(raw, FakeModel())

    assert list(features.columns) == [
        "text_length",
        "word_count",
        "rating",
        "prediction",
    ]
    assert features["word_count"].tolist() == [2, 3]
    assert features["rating"].tolist() == ["5", "1"]
    assert features["prediction"].tolist() == ["positive", "positive"]


def test_drifted_share_finds_the_share_metric():
    result = {"metrics": [{"value": 0.3}, {"value": {"count": 2.0, "share": 0.5}}]}
    assert drifted_share(result) == 0.5


def test_drifted_share_returns_none_when_missing():
    assert drifted_share({"metrics": [{"value": 0.1}]}) is None


def test_metric_label_handles_different_keys():
    assert metric_label({"metric_name": "A"}) == "A"
    assert metric_label({"metric_id": "B"}) == "B"
    assert metric_label({}) == "metric"
