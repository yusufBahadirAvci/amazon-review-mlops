import math
import re

from bs4 import Tag

# "a-star-4", "a-star-4-5" (4.5), "a-star-medium-4"; the same in every marketplace language
_STAR_CLASS = re.compile(r"^a-star-(?:[a-z]+-)?(\d)(?:-(\d))?$")

_NUMBER = r"(\d+(?:[.,]\d+)?)"
_LABEL_PATTERNS = (
    re.compile(_NUMBER + r"\s+out\s+of\s+5\b", re.IGNORECASE),  # 4.0 out of 5 stars
    re.compile(_NUMBER + r"\s+von\s+5\b", re.IGNORECASE),  # 4,0 von 5 Sternen
    re.compile(r"\b5\s+yıldız\s+üzerinden\s+" + _NUMBER, re.IGNORECASE),  # 5 yıldız üzerinden 4,0
)


def parse_rating(element: Tag | None) -> int | None:
    """Rating from a star icon: the CSS class first, the text label as a fallback.

    The result is not range-checked; callers decide what counts as a valid rating.
    """
    if element is None:
        return None
    for css_class in element.get("class") or []:
        match = _STAR_CLASS.match(css_class)
        if match:
            whole, fraction = match.groups()
            return _round_half_up(float(f"{whole}.{fraction or 0}"))
    label = element.select_one(".a-icon-alt") or element
    return parse_rating_label(label.get_text(" ", strip=True))


def parse_rating_label(label: str) -> int | None:
    """Rating from a label such as "4.0 out of 5 stars"; accepts dot and comma decimals."""
    label = label.replace("\xa0", " ")
    for pattern in _LABEL_PATTERNS:
        match = pattern.search(label)
        if match:
            return _round_half_up(float(match.group(1).replace(",", ".")))
    return None


def _round_half_up(value: float) -> int:
    # round() rounds 4.5 down to 4 (banker's rounding); a 4.5-star review should count as 5
    return math.floor(value + 0.5)
