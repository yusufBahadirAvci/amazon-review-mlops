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
    english, turkish = db.scalars(select(Review).order_by(Review.id)).all()
    assert english.product_id == "B000TEST01"
    assert english.marketplace == "amazon.com"
    assert english.source == ReviewSource.CRAWLER_HTML_FILE
    assert english.rating == 4
    assert english.title == "Solid build, average battery"
    assert english.text == "Works well for daily use. Battery lasts about a day."
    assert english.review_date == "Reviewed in the United States on March 3, 2025"
    assert turkish.rating == 3
    assert turkish.title == "Fiyatına göre iyi"
    assert len(english.content_hash) == 64


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
        assert "Ahmet Y." not in stored


def test_import_current_amazon_tr_layout(db, tr_reviews_html, crawler_settings):
    url = "https://www.amazon.com.tr/Seker-Portakali/dp/9750738608/ref=s9_acsd?_encoding=UTF8"

    result = import_html(tr_reviews_html, url, db, crawler_settings)

    assert (result.found, result.saved, result.skipped_invalid) == (3, 3, 0)
    turkish, english, german = db.scalars(select(Review).order_by(Review.id)).all()
    assert {review.product_id for review in (turkish, english, german)} == {"9750738608"}
    assert {review.marketplace for review in (turkish, english, german)} == {"amazon.com.tr"}
    assert (turkish.title, turkish.text, turkish.rating) == (
        "Güzel",
        "Ah zeze.. Bu bilinçle okumak çok güzeldi",
        5,
    )
    assert turkish.review_date == "Türkiye’de 19 Ekim 2026 tarihinde değerlendirildi"
    # Paragraphs are joined; translated reviews keep their original language
    assert english.text == (
        "I didn’t like the book because it start from page 18 Before that all pages are missing👎"
    )
    assert (english.title, english.rating) == ("Missing page", 1)
    assert german.title == "Schöne Geschichte, herzzerreißend"
    assert german.text.startswith("Das Buch ist so schön geschrieben.")
    for review in (turkish, english, german):
        # Hidden accessibility text and translation widgets must not leak into the review
        assert "Full content visible" not in review.text
        assert "çevir" not in review.text


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
        ("https://www.amazon.com.tr/dp/B0ABCDEFGH?th=1", "amazon.com.tr", "B0ABCDEFGH"),
        ("http://amazon.de/gp/product/B0ABCDEFGH/", "amazon.de", "B0ABCDEFGH"),
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
    ],
)
def test_rejects_non_amazon_urls(url):
    with pytest.raises(InvalidCrawlInputError):
        parse_amazon_url(url)
