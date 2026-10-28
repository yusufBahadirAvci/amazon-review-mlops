import pytest
from bs4 import BeautifulSoup

from app.crawler.rating import parse_rating, parse_rating_label


def star_icon(css_class: str, label: str) -> BeautifulSoup:
    html = f'<i class="a-icon a-icon-star {css_class}"><span class="a-icon-alt">{label}</span></i>'
    return BeautifulSoup(html, "lxml").i


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("4.0 out of 5 stars", 4),
        ("3.5 out of 5 stars", 4),
        ("5 yıldız üzerinden 4,0", 4),
        ("5 yıldız üzerinden 2,4", 2),
        ("4,0 von 5 Sternen", 4),
        ("4,5\xa0von 5 Sternen", 5),
        ("no rating here", None),
    ],
)
def test_rating_from_label(label, expected):
    assert parse_rating_label(label) == expected


@pytest.mark.parametrize(
    ("css_class", "expected"),
    [("a-star-4", 4), ("a-star-4-5", 5), ("a-star-medium-3", 3), ("a-star-1", 1)],
)
def test_css_class_takes_precedence_over_label(css_class, expected):
    # The label deliberately disagrees, to prove the class wins
    assert parse_rating(star_icon(css_class, "2.0 out of 5 stars")) == expected


def test_falls_back_to_label_without_star_class():
    assert parse_rating(star_icon("a-icon-star-small", "5 yıldız üzerinden 3,0")) == 3


def test_missing_element_has_no_rating():
    assert parse_rating(None) is None
