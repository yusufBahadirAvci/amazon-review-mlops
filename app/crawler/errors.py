class CrawlerError(Exception):
    """Crawling failed for a reason other than bad input or being blocked."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class CrawlerBlockedError(CrawlerError):
    """Amazon refused the request (HTTP 403/503 or a captcha page)."""


class InvalidCrawlInputError(CrawlerError):
    """The URL or uploaded file cannot be crawled."""
