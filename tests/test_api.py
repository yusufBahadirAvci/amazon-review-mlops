import requests

PRODUCT_URL = "https://www.amazon.com/product-reviews/B000TEST01"


def fake_response(status_code: int, body: bytes = b"") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    return response


def upload(api_client, html: bytes, source_url: str | None = PRODUCT_URL):
    data = {"source_url": source_url} if source_url else {}
    return api_client.post(
        "/api/crawler/html", files={"file": ("reviews.html", html, "text/html")}, data=data
    )


def test_crawl_url(monkeypatch, api_client, reviews_html):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: fake_response(200, reviews_html))

    response = api_client.post("/api/crawler/url", json={"url": PRODUCT_URL})

    assert response.status_code == 200
    body = response.json()
    assert (body["found"], body["saved"], body["skipped_invalid"]) == (3, 2, 1)


def test_crawl_url_blocked_returns_503(monkeypatch, api_client):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: fake_response(503))

    response = api_client.post("/api/crawler/url", json={"url": PRODUCT_URL})

    assert response.status_code == 503
    assert "/api/crawler/html" in response.json()["hint"]


def test_crawl_url_rejects_non_amazon_url(api_client):
    response = api_client.post("/api/crawler/url", json={"url": "https://example.com/dp/X"})

    assert response.status_code == 400
    assert set(response.json()) == {"error", "hint"}


def test_crawl_url_rejects_turkish_marketplace(api_client):
    response = api_client.post(
        "/api/crawler/url", json={"url": "https://www.amazon.com.tr/dp/B0ABCDEFGH"}
    )

    assert response.status_code == 400
    assert "amazon.com.tr" in response.json()["error"]
    assert "amazon.co.uk" in response.json()["hint"]


def test_import_html_rejects_turkish_source_url(api_client, reviews_html):
    response = upload(
        api_client, reviews_html, source_url="https://www.amazon.com.tr/dp/B0ABCDEFGH"
    )

    assert response.status_code == 400


def test_crawl_url_missing_body_returns_400(api_client):
    response = api_client.post("/api/crawler/url", json={})

    assert response.status_code == 400
    assert "url" in response.json()["error"]


def test_import_html(api_client, reviews_html):
    first = upload(api_client, reviews_html)
    second = upload(api_client, reviews_html)

    assert first.status_code == 200
    assert first.json()["saved"] == 2
    assert second.json()["skipped_duplicate"] == 2


def test_import_html_without_source_url(api_client, reviews_html):
    response = upload(api_client, reviews_html, source_url=None)

    assert response.status_code == 200
    assert response.json()["saved"] == 2


def test_import_captcha_page_returns_400(api_client, captcha_html):
    response = upload(api_client, captcha_html)

    assert response.status_code == 400
    assert "captcha" in response.json()["error"]


def test_import_empty_file_returns_400(api_client):
    assert upload(api_client, b"   ").status_code == 400


def test_list_reviews_filters(api_client, reviews_html):
    upload(api_client, reviews_html)

    all_reviews = api_client.get("/api/reviews").json()
    from_files = api_client.get(
        "/api/reviews", params={"source": "CRAWLER_HTML_FILE", "marketplace": "amazon.com"}
    ).json()
    live = api_client.get("/api/reviews", params={"source": "CRAWLER_LIVE"}).json()
    other_marketplace = api_client.get("/api/reviews", params={"marketplace": "amazon.de"}).json()

    assert len(all_reviews) == len(from_files) == 2
    assert live == other_marketplace == []
    assert {review["rating"] for review in from_files} == {3, 4}
    assert "content_hash" not in from_files[0]


def test_list_reviews_rejects_unknown_source(api_client):
    response = api_client.get("/api/reviews", params={"source": "SOMETHING"})

    assert response.status_code == 400
