from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import CrawlerSettings, ModelSettings, get_crawler_settings, get_model_settings
from app.db import Base, get_session
from app.main import app
from app.ml_model import LoadedModel, ModelStore, get_model_store

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def reviews_html() -> bytes:
    return (FIXTURES / "amazon-reviews.html").read_bytes()


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
def model_settings() -> ModelSettings:
    return ModelSettings(_env_file=None)


class FakeSentimentModel:
    """Behaves like the MLflow pyfunc model: DataFrame with "text" in, numpy labels out."""

    def __init__(self) -> None:
        self.inputs: list[pd.DataFrame] = []

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        self.inputs.append(data)
        return np.array([self._label(text) for text in data["text"]])

    @staticmethod
    def _label(text: str) -> str:
        text = text.lower()
        if "love" in text or "great" in text:
            return "positive"
        if "broke" in text or "terrible" in text:
            return "negative"
        return "neutral"


class FakeModelLoader:
    """Stands in for MLflow: hands out fake models with increasing version numbers."""

    def __init__(self, model: object) -> None:
        self.model = model
        self.calls = 0
        self.attempts = 0
        self.fail = False
        # Fail only the next N attempts, with this error
        self.fail_times = 0
        self.error: Exception = ConnectionError("MLflow is not reachable")

    def __call__(self, settings: ModelSettings) -> LoadedModel:
        self.attempts += 1
        if self.fail or self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        self.calls += 1
        return LoadedModel(self.model, str(self.calls), f"models:/sentiment-model/{self.calls}")


@pytest.fixture
def model_loader() -> FakeModelLoader:
    return FakeModelLoader(FakeSentimentModel())


@pytest.fixture
def model_store(model_loader: FakeModelLoader) -> ModelStore:
    """An empty store; call model_store.load(...) in a test to make the model available."""
    return ModelStore(loader=model_loader)


@pytest.fixture
def api_client(
    db: Session,
    crawler_settings: CrawlerSettings,
    model_settings: ModelSettings,
    model_store: ModelStore,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_crawler_settings] = lambda: crawler_settings
    app.dependency_overrides[get_model_settings] = lambda: model_settings
    app.dependency_overrides[get_model_store] = lambda: model_store
    # Not used as a context manager, so the startup hook never touches PostgreSQL
    yield TestClient(app)
    app.dependency_overrides.clear()
