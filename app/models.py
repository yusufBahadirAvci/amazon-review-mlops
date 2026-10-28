from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, Enum, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ReviewSource(StrEnum):
    DATASET = "DATASET"
    CRAWLER_LIVE = "CRAWLER_LIVE"
    CRAWLER_HTML_FILE = "CRAWLER_HTML_FILE"


class Review(Base):
    """A product review. Reviewer names are deliberately not stored."""

    __tablename__ = "reviews"
    __table_args__ = (CheckConstraint("rating BETWEEN 1 AND 5", name="ck_reviews_rating_range"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # ASIN parsed from the product URL; unknown for HTML files uploaded without a URL
    product_id: Mapped[str | None] = mapped_column(String(10), index=True)
    marketplace: Mapped[str | None] = mapped_column(String(32), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    rating: Mapped[int] = mapped_column(SmallInteger)
    # Kept as displayed ("Reviewed in ... on March 3, 2025"); the format depends on the marketplace
    review_date: Mapped[str | None] = mapped_column(String(255))
    # Stored as VARCHAR so new sources do not need a PostgreSQL enum migration
    source: Mapped[ReviewSource] = mapped_column(
        Enum(ReviewSource, native_enum=False, length=32), index=True
    )
    source_url: Mapped[str | None] = mapped_column(Text)
    # SHA-256 of product_id + normalized text, used to skip duplicates
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    crawled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
