"""Tests for Ceylin's data pipeline: data/download_data.py and data/validate_data.py.

No network, MLflow or database needed.
"""

import pandas as pd
import pytest

from data import download_data
from data.download_data import clean_reviews, rating_to_label, sample_balanced
from data.validate_data import validate_reviews


@pytest.mark.parametrize(
    ("rating", "label"),
    [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_rating_to_label_follows_contract(rating, label):
    assert rating_to_label(rating) == label


def test_clean_reviews_drops_bad_rows():
    raw = pd.DataFrame(
        {
            "rating": [5.0, 1.0, 3.0, 4.0, 2.0, 5.0, None],
            "text": [
                "Great<br />product",
                "n/a",
                "   ",
                "Great<br />product",
                "NULL",
                "Works fine",
                "No rating",
            ],
        }
    )
    cleaned = clean_reviews(raw)

    # n/a, empty, NULL, the duplicate and the row without a rating are removed
    assert cleaned["text"].tolist() == ["Great product", "Works fine"]
    assert cleaned["rating"].tolist() == [5, 5]
    assert cleaned["label"].tolist() == ["positive", "positive"]


def test_sample_balanced_takes_same_number_per_class(monkeypatch):
    monkeypatch.setattr(download_data, "SAMPLES_PER_CLASS_PER_CATEGORY", 2)
    df = pd.DataFrame(
        {
            "text": [f"review {i}" for i in range(9)],
            "label": ["positive"] * 5 + ["neutral"] * 3 + ["negative"] * 1,
        }
    )
    counts = sample_balanced(df)["label"].value_counts().to_dict()
    assert counts == {"positive": 2, "neutral": 2, "negative": 1}


def _valid_reviews(n: int = 1200) -> pd.DataFrame:
    return pd.DataFrame(
        {"text": ["good product"] * n, "rating": [5] * n, "label": ["positive"] * n}
    )


def test_validate_reviews_accepts_valid_data():
    assert validate_reviews(_valid_reviews())


def test_validate_reviews_rejects_rating_out_of_range():
    df = _valid_reviews()
    df.loc[0, "rating"] = 7
    assert not validate_reviews(df)


def test_validate_reviews_rejects_unknown_label():
    df = _valid_reviews()
    df.loc[0, "label"] = "great"
    assert not validate_reviews(df)
