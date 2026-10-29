import hashlib
import time
import unicodedata
from dataclasses import dataclass, field

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import CrawlerSettings, get_crawler_settings
from app.crawler.errors import CrawlerBlockedError, CrawlerError, InvalidCrawlInputError
from app.crawler.parser import BLOCKED_HINT, ParsedReview, parse_amazon_url, parse_reviews
from app.models import Review, ReviewSource


@dataclass
class CrawlResult:
    found: int = 0
    saved: int = 0
    skipped_invalid: int = 0
    skipped_duplicate: int = 0
    # Translated reviews and reviews from other countries (we store English originals only)
    skipped_foreign: int = 0
    warnings: list[str] = field(default_factory=list)


def crawl_url(url: str, db: Session, settings: CrawlerSettings | None = None) -> CrawlResult:
    """Download one Amazon reviews page and save its reviews."""
    settings = settings or get_crawler_settings()
    target = parse_amazon_url(url)
    reviews = parse_reviews(fetch_page(target.url, settings), settings)
    return _save_reviews(
        reviews,
        db=db,
        settings=settings,
        source=ReviewSource.CRAWLER_LIVE,
        source_url=target.url,
        product_id=target.product_id,
        marketplace=target.marketplace,
    )


def fetch_page(url: str, settings: CrawlerSettings) -> bytes:
    """Download a page with the configured user agent; raises on blocks and HTTP errors."""
    # Be polite: never send requests back to back
    time.sleep(settings.delay_seconds)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": settings.user_agent},
            timeout=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise CrawlerError(
            f"Request to {url} failed: {exc}",
            hint="Check the URL and your network connection.",
        ) from exc

    if response.status_code in (403, 503):
        raise CrawlerBlockedError(f"Amazon returned HTTP {response.status_code}", hint=BLOCKED_HINT)
    if not response.ok:
        raise CrawlerError(
            f"Amazon returned HTTP {response.status_code}",
            hint="Check that the URL points to an existing reviews page.",
        )
    return response.content


def import_html(
    html: str | bytes,
    source_url: str | None,
    db: Session,
    settings: CrawlerSettings | None = None,
) -> CrawlResult:
    """Save the reviews from a reviews page that was saved in a browser."""
    settings = settings or get_crawler_settings()
    target = parse_amazon_url(source_url) if source_url else None
    try:
        reviews = parse_reviews(html, settings)
    except CrawlerBlockedError as exc:
        raise InvalidCrawlInputError(
            "The uploaded file is a captcha page, not a reviews page",
            hint="Solve the captcha in your browser, reload the reviews page, then save it again.",
        ) from exc
    return _save_reviews(
        reviews,
        db=db,
        settings=settings,
        source=ReviewSource.CRAWLER_HTML_FILE,
        source_url=target.url if target else None,
        product_id=target.product_id if target else None,
        marketplace=target.marketplace if target else None,
    )


def content_hash(product_id: str | None, text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(f"{product_id or ''}|{normalized}".encode()).hexdigest()


def _save_reviews(
    reviews: list[ParsedReview],
    *,
    db: Session,
    settings: CrawlerSettings,
    source: ReviewSource,
    source_url: str | None,
    product_id: str | None,
    marketplace: str | None,
) -> CrawlResult:
    result = CrawlResult(found=len(reviews))
    if not reviews:
        result.warnings.append("No reviews found on the page; the CSS selectors may be outdated.")
    limit = settings.max_reviews_per_run
    if len(reviews) > limit:
        result.warnings.append(f"Only the first {limit} of {len(reviews)} reviews were processed.")
        reviews = reviews[:limit]

    seen: set[str] = set()
    for number, review in enumerate(reviews, start=1):
        if review.skip_reason is not None:
            result.skipped_foreign += 1
            result.warnings.append(f"Review {number}: {review.skip_reason}, skipped.")
            continue
        if review.text is None:
            result.skipped_invalid += 1
            result.warnings.append(
                f"Review {number}: no element matched the text selector "
                f"({settings.text_selector!r}), skipped. The page layout may have changed."
            )
            continue
        if not review.text:
            result.skipped_invalid += 1
            result.warnings.append(f"Review {number}: empty text, skipped.")
            continue
        if review.rating is None or not 1 <= review.rating <= 5:
            result.skipped_invalid += 1
            result.warnings.append(f"Review {number}: missing or invalid rating, skipped.")
            continue

        digest = content_hash(product_id, review.text)
        already_stored = db.scalar(select(Review.id).where(Review.content_hash == digest))
        if digest in seen or already_stored is not None:
            result.skipped_duplicate += 1
            continue
        seen.add(digest)

        db.add(
            Review(
                product_id=product_id,
                marketplace=marketplace,
                title=review.title,
                text=review.text,
                rating=review.rating,
                review_date=review.review_date,
                source=source,
                source_url=source_url,
                content_hash=digest,
            )
        )
        result.saved += 1

    db.commit()
    return result
