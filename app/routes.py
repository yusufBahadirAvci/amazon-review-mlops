from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import CrawlerSettings, get_crawler_settings
from app.crawler import CrawlResult, InvalidCrawlInputError, crawl_url, import_html
from app.db import get_session
from app.models import Review, ReviewSource

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

router = APIRouter(prefix="/api")

SessionDep = Annotated[Session, Depends(get_session)]
CrawlerSettingsDep = Annotated[CrawlerSettings, Depends(get_crawler_settings)]


class CrawlUrlRequest(BaseModel):
    url: str


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: str | None
    marketplace: str | None
    title: str | None
    text: str
    rating: int
    review_date: str | None
    source: ReviewSource
    source_url: str | None
    crawled_at: datetime


@router.post("/crawler/url")
def crawl_from_url(
    request: CrawlUrlRequest, db: SessionDep, settings: CrawlerSettingsDep
) -> CrawlResult:
    return crawl_url(request.url, db, settings)


@router.post("/crawler/html")
def crawl_from_html(
    file: Annotated[UploadFile, File()],
    db: SessionDep,
    settings: CrawlerSettingsDep,
    source_url: Annotated[str | None, Form()] = None,
) -> CrawlResult:
    html = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(html) > MAX_UPLOAD_BYTES:
        raise InvalidCrawlInputError(
            "The uploaded file is larger than 5 MB",
            hint="Upload a single reviews page saved as HTML only.",
        )
    if not html.strip():
        raise InvalidCrawlInputError(
            "The uploaded file is empty", hint="Save the reviews page as HTML and upload it."
        )
    return import_html(html, source_url or None, db, settings)


@router.get("/reviews")
def list_reviews(
    db: SessionDep,
    source: ReviewSource | None = None,
    marketplace: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ReviewOut]:
    query = select(Review).order_by(Review.id.desc()).limit(limit)
    if source is not None:
        query = query.where(Review.source == source)
    if marketplace is not None:
        query = query.where(Review.marketplace == marketplace)
    return [ReviewOut.model_validate(review) for review in db.scalars(query)]
