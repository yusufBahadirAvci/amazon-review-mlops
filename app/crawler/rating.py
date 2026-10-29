import math
import re

from bs4 import Tag

# "a-star-4", "a-star-4-5" (4.5), "a-star-medium-4"; independent of the page language
_STAR_CLASS = re.compile(r"^a-star-(?:[a-z]+-)?(\d)(?:-(\d))?$")

# "4.0 out of 5 stars"
_ENGLISH_LABEL = re.compile(r"(\d+(?:\.\d+)?)\s+out\s+of\s+5\b", re.IGNORECASE)


def parse_rating(element: Tag | None) -> int | None:
    """Rating from a star icon: the CSS class first, the English text label as a fallback.

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
    """Rating from an English label such as "4.0 out of 5 stars"."""
    match = _ENGLISH_LABEL.search(label.replace("\xa0", " "))
    if match:
        return _round_half_up(float(match.group(1)))
    return None


def _round_half_up(value: float) -> int:
    # round() rounds 4.5 down to 4 (banker's rounding); a 4.5-star review should count as 5
    return math.floor(value + 0.5)
