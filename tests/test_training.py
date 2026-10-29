"""Tests for Ceylin's training code: training/train.py.

Uses a tiny toy dataset; no MLflow server or real data needed.
"""

import pandas as pd
import pytest

from training import train
from training.train import build_pipeline

TOY_TEXTS = {
    "positive": [
        "great product, love it",
        "excellent quality, very happy",
        "works perfectly, highly recommend",
    ],
    "neutral": [
        "it is okay, nothing special",
        "average quality, does the job",
        "fine but not great",
    ],
    "negative": [
        "terrible, broke after a day",
        "awful quality, waste of money",
        "does not work, very disappointed",
    ],
}


def _toy_data():
    rows = [(text, label) for label, texts in TOY_TEXTS.items() for text in texts] * 5
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df[["text"]], df["label"]


@pytest.mark.parametrize("classifier", ["logreg", "linearsvc"])
def test_pipeline_takes_text_column_and_returns_labels(classifier):
    """Project contract: DataFrame with a "text" column in, label strings out."""
    params = {
        "classifier": classifier,
        "C": 1.0,
        "ngram_max": 2,
        "min_df": 1,
        "max_features": 1000,
        "sublinear_tf": True,
    }
    x, y = _toy_data()
    model = build_pipeline(params).fit(x, y)

    new_reviews = pd.DataFrame({"text": ["love it, excellent", "terrible, waste of money"]})
    predictions = list(model.predict(new_reviews))

    assert set(predictions) <= {"negative", "neutral", "positive"}
    assert predictions == ["positive", "negative"]


def test_data_version_reads_md5_from_dvc_file(tmp_path, monkeypatch):
    dvc_file = tmp_path / "amazon_reviews.csv.dvc"
    dvc_file.write_text(
        "outs:\n- md5: abc123\n  size: 10\n  hash: md5\n  path: amazon_reviews.csv\n"
    )
    monkeypatch.setattr(train, "DVC_FILE", dvc_file)
    assert train.data_version() == "abc123"
