from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import CrawlerSettings, get_crawler_settings
from app.db import Base, get_session
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def reviews_html() -> bytes:
    return (FIXTURES / "amazon-reviews.html").read_bytes()


@pytest.fixture
def tr_reviews_html() -> bytes:
    """Current amazon.com.tr product-page layout, trimmed and anonymized."""
    return (FIXTURES / "amazon-tr-reviews.html").read_bytes()


@pytest.fixture
def captcha_html() -> bytes:
    return (FIXTURES / "amazon-captcha.html").read_bytes()


@pytest.fixture
def crawler_settings() -> CrawlerSettings:
    # Ignore .env so local overrides cannot change test results; no delay between requests
    return CrawlerSettings(_env_file=None, delay_seconds=0)


@pytest.fixture
def db() -> Iterator[Session]:
    # StaticPool keeps one in-memory database shared by the TestClient's worker threads
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session
    engine.dispose()


@pytest.fixture
def api_client(db: Session, crawler_settings: CrawlerSettings) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_crawler_settings] = lambda: crawler_settings
    # Not used as a context manager, so the startup hook never touches PostgreSQL
    yield TestClient(app)
    app.dependency_overrides.clear()
