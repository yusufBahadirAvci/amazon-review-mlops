import pytest
import requests
from sqlalchemy import select

from app.crawler import CrawlerBlockedError, InvalidCrawlInputError, crawl_url, import_html
from app.crawler.parser import parse_amazon_url, parse_reviews
from app.models import Review, ReviewSource

PRODUCT_URL = "https://www.amazon.com/product-reviews/B000TEST01/ref=cm_cr_dp"


def fake_response(status_code: int, body: bytes = b"") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    return response


def test_import_saves_valid_reviews_and_skips_empty_body(db, reviews_html, crawler_settings):
    result = import_html(reviews_html, PRODUCT_URL, db, crawler_settings)

    assert (result.found, result.saved, result.skipped_invalid, result.skipped_duplicate) == (
        3,
        2,
        1,
        0,
    )
    first, second = db.scalars(select(Review).order_by(Review.id)).all()
    assert first.product_id == "B000TEST01"
    assert first.marketplace == "amazon.com"
    assert first.source == ReviewSource.CRAWLER_HTML_FILE
    assert first.rating == 4
    assert first.title == "Solid build, average battery"
    assert first.text == "Works well for daily use. Battery lasts about a day."
    assert first.review_date == "Reviewed in the United States on March 3, 2025"
    assert len(first.content_hash) == 64
    assert (second.rating, second.title) == (3, "Good for the price")
    assert second.text == "Shipping was fast but the box was a bit dented."
    assert second.review_date == "Reviewed in the United States on January 12, 2025"


def test_second_import_skips_duplicates(db, reviews_html, crawler_settings):
    import_html(reviews_html, PRODUCT_URL, db, crawler_settings)

    result = import_html(reviews_html, PRODUCT_URL, db, crawler_settings)

    assert (result.saved, result.skipped_invalid, result.skipped_duplicate) == (0, 1, 2)
    assert len(db.scalars(select(Review)).all()) == 2


def test_reviewer_names_are_not_stored(db, reviews_html, crawler_settings):
    import_html(reviews_html, PRODUCT_URL, db, crawler_settings)

    for review in db.scalars(select(Review)):
        stored = " ".join(str(value) for value in vars(review).values())
        assert "Jane Doe" not in stored
        assert "Alex P." not in stored


def current_layout_review(text: str, *, lang: str = "en-US", extra: str = "") -> str:
    """One review in the current product-page layout (reviewTitle, reviewRichContentContainer)."""
    return f"""
        <div data-hook="review">
            <i data-hook="review-star-rating" class="a-icon a-icon-star a-star-2">
                <span class="a-icon-alt">2.0 out of 5 stars</span>
            </i>
            <a href="#"><h5 data-hook="reviewTitle" lang="{lang}">Pages missing</h5></a>
            <span data-hook="review-date">Reviewed in the United States on May 2, 2026</span>
            <div data-hook="reviewText">
                <div class="a-teaser-describedby-expanded a-hidden">
                    Full content visible, double tap to read brief content.
                </div>
                <div data-hook="reviewRichContentContainer" lang="{lang}">
                    <p><span>{text}</span></p><p><span>Second paragraph.</span></p>
                </div>
            </div>
            {extra}
        </div>
    """


def test_import_current_layout(db, crawler_settings):
    html = current_layout_review("The book starts at page 18.")

    result = import_html(html, PRODUCT_URL, db, crawler_settings)

    assert (result.found, result.saved, result.skipped_foreign) == (1, 1, 0)
    review = db.scalars(select(Review)).one()
    assert (review.title, review.rating) == ("Pages missing", 2)
    # Paragraphs are joined; hidden accessibility text must not leak into the review
    assert review.text == "The book starts at page 18. Second paragraph."


@pytest.mark.parametrize(
    ("html", "reason"),
    [
        (
            current_layout_review(
                "Translated text.",
                extra='<div data-hook="translateButton"><a href="#">Translate review</a></div>',
            ),
            "translated",
        ),
        (
            current_layout_review(
                "Shown as translated.",
                extra='<div data-hook="translationInfo">Translated from French by Amazon</div>',
            ),
            "translated",
        ),
        (
            '<div id="cm-cr-global-review-list"><h3>Top reviews from other countries</h3>'
            + current_layout_review("Written in another country.")
            + "</div>",
            "other countries",
        ),
        (current_layout_review("Texto en otro idioma.", lang="es-MX"), "not in English"),
    ],
)
def test_translated_and_foreign_reviews_are_skipped(db, crawler_settings, html, reason):
    result = import_html(html, PRODUCT_URL, db, crawler_settings)

    assert (result.found, result.saved, result.skipped_foreign) == (1, 0, 1)
    assert reason in result.warnings[0]
    assert db.scalars(select(Review)).all() == []


def test_foreign_reviews_do_not_block_local_ones(db, reviews_html, crawler_settings):
    foreign = '<div id="cm-cr-global-review-list">' + current_layout_review("Abroad.") + "</div>"
    html = reviews_html.decode().replace("</body>", foreign + "</body>")

    result = import_html(html, PRODUCT_URL, db, crawler_settings)

    assert (result.found, result.saved, result.skipped_invalid, result.skipped_foreign) == (
        4,
        2,
        1,
        1,
    )


def test_missing_text_element_is_reported(db, crawler_settings):
    html = """
        <div data-hook="review">
            <i data-hook="review-star-rating" class="a-icon a-star-4"></i>
            <div data-hook="some-future-body">Text in a layout we do not know yet</div>
        </div>
    """

    result = import_html(html, None, db, crawler_settings)

    assert (result.found, result.saved, result.skipped_invalid) == (1, 0, 1)
    assert "text selector" in result.warnings[0]


def test_captcha_page_is_detected(captcha_html, crawler_settings):
    with pytest.raises(CrawlerBlockedError) as error:
        parse_reviews(captcha_html, crawler_settings)

    assert "/api/crawler/html" in error.value.hint


def test_captcha_text_without_form_is_detected(crawler_settings):
    html = "<html><body><h4>Enter the characters you see below</h4></body></html>"

    with pytest.raises(CrawlerBlockedError):
        parse_reviews(html, crawler_settings)


def test_uploaded_captcha_page_is_invalid_input(db, captcha_html, crawler_settings):
    with pytest.raises(InvalidCrawlInputError):
        import_html(captcha_html, None, db, crawler_settings)


@pytest.mark.parametrize("status_code", [403, 503])
def test_live_crawl_blocked_by_status(monkeypatch, db, crawler_settings, status_code):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: fake_response(status_code))

    with pytest.raises(CrawlerBlockedError):
        crawl_url(PRODUCT_URL, db, crawler_settings)


def test_live_crawl_blocked_by_captcha_page(monkeypatch, db, captcha_html, crawler_settings):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: fake_response(200, captcha_html))

    with pytest.raises(CrawlerBlockedError):
        crawl_url(PRODUCT_URL, db, crawler_settings)


def test_live_crawl_sends_only_the_configured_user_agent(
    monkeypatch, db, reviews_html, crawler_settings
):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return fake_response(200, reviews_html)

    monkeypatch.setattr(requests, "get", fake_get)

    result = crawl_url(PRODUCT_URL, db, crawler_settings)

    assert result.saved == 2
    assert calls == [
        (
            PRODUCT_URL,
            {
                "headers": {"User-Agent": crawler_settings.user_agent},
                "timeout": crawler_settings.timeout_seconds,
            },
        )
    ]
    assert {review.source for review in db.scalars(select(Review))} == {ReviewSource.CRAWLER_LIVE}


@pytest.mark.parametrize(
    ("url", "marketplace", "product_id"),
    [
        ("https://www.amazon.com/dp/B0ABCDEFGH?th=1", "amazon.com", "B0ABCDEFGH"),
        ("http://amazon.ca/gp/product/B0ABCDEFGH/", "amazon.ca", "B0ABCDEFGH"),
        ("https://www.amazon.com.au/product-reviews/B0ABCDEFGH", "amazon.com.au", "B0ABCDEFGH"),
        ("https://www.amazon.in/dp/B0ABCDEFGH", "amazon.in", "B0ABCDEFGH"),
        ("https://www.amazon.co.uk/some-product", "amazon.co.uk", None),
    ],
)
def test_parse_amazon_url(url, marketplace, product_id):
    parsed = parse_amazon_url(url)

    assert (parsed.marketplace, parsed.product_id) == (marketplace, product_id)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://www.amazon.com/dp/B0ABCDEFGH",
        "https://www.example.com/dp/B0ABCDEFGH",
        "https://amazon.com.evil.example/dp/B0ABCDEFGH",
        "https://www.amazon.com@evil.example/dp/B0ABCDEFGH",
        "not a url",
        # Non-English marketplaces
        "https://www.amazon.com.tr/dp/B0ABCDEFGH",
        "https://www.amazon.de/dp/B0ABCDEFGH",
        "https://www.amazon.fr/dp/B0ABCDEFGH",
        "https://www.amazon.co.jp/dp/B0ABCDEFGH",
    ],
)
def test_rejects_unsupported_urls(url):
    with pytest.raises(InvalidCrawlInputError):
        parse_amazon_url(url)
