import copy
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

from app.config import CrawlerSettings
from app.crawler.errors import CrawlerBlockedError, InvalidCrawlInputError
from app.crawler.rating import parse_rating

# Team decision: English reviews only, so only English-language marketplaces are crawled
ENGLISH_MARKETPLACES = frozenset(
    {"amazon.com", "amazon.co.uk", "amazon.ca", "amazon.com.au", "amazon.in"}
)

_ASIN = re.compile(r"/(?:dp|gp/product|product-reviews|gp/aw/d)/([A-Z0-9]{10})(?=[/?#]|$)", re.I)

CAPTCHA_TEXT = "Enter the characters you see below"
BLOCKED_HINT = (
    "Amazon blocked the automated request. Open the reviews page in your browser, "
    "save it (Ctrl+S) and upload it to POST /api/crawler/html instead."
)


@dataclass(frozen=True)
class AmazonUrl:
    url: str
    marketplace: str
    product_id: str | None


@dataclass(frozen=True)
class ParsedReview:
    title: str | None
    # None when the text selector matched nothing, "" when the review body is blank
    text: str | None
    rating: int | None
    review_date: str | None
    # Set when the review must not be stored (translated or from another country)
    skip_reason: str | None = None


def parse_amazon_url(url: str) -> AmazonUrl:
    """Validate an http(s) URL on an English Amazon marketplace; extract marketplace and ASIN."""
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise InvalidCrawlInputError(
            "URL must start with http:// or https://",
            hint="Example: https://www.amazon.com/product-reviews/B0EXAMPLE1",
        )
    marketplace = (parts.hostname or "").lower().removeprefix("www.")
    if marketplace not in ENGLISH_MARKETPLACES:
        raise InvalidCrawlInputError(
            f"Not a supported Amazon marketplace: {parts.hostname or url}",
            hint="Only English marketplaces are supported: "
            + ", ".join(sorted(ENGLISH_MARKETPLACES)),
        )
    match = _ASIN.search(parts.path)
    return AmazonUrl(url, marketplace, match.group(1).upper() if match else None)


def parse_reviews(html: str | bytes, settings: CrawlerSettings) -> list[ParsedReview]:
    """Parse every review on the page; raises CrawlerBlockedError for captcha pages."""
    soup = BeautifulSoup(html, "lxml")
    if is_blocked_page(soup):
        raise CrawlerBlockedError("Amazon returned a captcha page", hint=BLOCKED_HINT)
    return [_parse_review(element, settings) for element in soup.select(settings.review_selector)]


def is_blocked_page(soup: BeautifulSoup) -> bool:
    if soup.select_one("form[action*=validateCaptcha]") is not None:
        return True
    return CAPTCHA_TEXT in _normalize(soup.get_text(" "))


def _parse_review(element: Tag, settings: CrawlerSettings) -> ParsedReview:
    body = element.select_one(settings.text_selector)
    return ParsedReview(
        title=_title(element.select_one(settings.title_selector)) or None,
        text=_body(body),
        rating=parse_rating(element.select_one(settings.rating_selector)),
        review_date=_text(element.select_one(settings.date_selector)) or None,
        skip_reason=_skip_reason(element, body, settings),
    )


def _skip_reason(element: Tag, body: Tag | None, settings: CrawlerSettings) -> str | None:
    if element.css.closest(settings.foreign_section_selector) is not None:
        return "from the 'reviews from other countries' section"
    if element.select_one(settings.translated_selector) is not None:
        return "marked as translated"
    lang = (body.get("lang") or "") if body is not None else ""
    if lang and not lang.lower().startswith("en"):
        return f"not in English (lang={lang})"
    return None


def _title(element: Tag | None) -> str:
    if element is None:
        return ""
    # On amazon.com the star icon sits inside the title link; drop it so "4.0 out of 5 stars"
    # does not end up in the title
    element = copy.copy(element)
    for icon in element.select("i, .a-icon-alt, .a-letter-space"):
        icon.decompose()
    return _text(element)


def _body(element: Tag | None) -> str | None:
    if element is None:
        return None
    # Long reviews are wrapped in an expander whose header adds a "Read more" link
    return _text(element.select_one(".a-expander-content") or element)


def _text(element: Tag | None) -> str:
    return _normalize(element.get_text(" ")) if element is not None else ""


def _normalize(text: str) -> str:
    # str.split() also splits on non-breaking spaces
    return " ".join(text.split())
