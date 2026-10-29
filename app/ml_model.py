"""Loads the sentiment model from the MLflow Model Registry and keeps it in memory."""

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from app.config import ModelSettings

logger = logging.getLogger(__name__)

MLFLOW_CHECK_TIMEOUT_SECONDS = 5

_ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/]+)$")
_VERSION_URI = re.compile(r"^models:/(?P<name>[^/@]+)/(?P<version>\d+)$")


class ModelUnavailableError(Exception):
    """The model is not loaded, or (re)loading it failed."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class LoadedModel:
    # mlflow.pyfunc.PyFuncModel in production; anything with predict(DataFrame) in tests
    model: Any
    version: str
    uri: str


def check_mlflow_reachable(tracking_uri: str) -> None:
    """Fail fast when the MLflow server is down.

    Without this, the MLflow client retries with backoff for about 90 seconds,
    which blocks app startup and POST /api/model/reload.
    """
    if not tracking_uri.startswith(("http://", "https://")):
        return
    try:
        requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=MLFLOW_CHECK_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise ConnectionError(f"MLflow is not reachable at {tracking_uri}") from exc


def load_from_mlflow(settings: ModelSettings) -> LoadedModel:
    check_mlflow_reachable(settings.mlflow_tracking_uri)
    # Imported here so the API and the tests start without loading MLflow
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = settings.model_uri
    version = "unknown"
    if match := _ALIAS_URI.match(uri):
        version = MlflowClient().get_model_version_by_alias(match["name"], match["alias"]).version
        # Load the exact version the alias points to, so the reported version always matches
        # the model in memory, even if the alias moves while we are loading
        uri = f"models:/{match['name']}/{version}"
    elif match := _VERSION_URI.match(uri):
        version = match["version"]
    return LoadedModel(model=mlflow.pyfunc.load_model(uri), version=str(version), uri=uri)


class ModelStore:
    """Holds the current model; a failed reload keeps the previous model serving."""

    def __init__(self, loader: Callable[[ModelSettings], LoadedModel] = load_from_mlflow) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self.current: LoadedModel | None = None

    def load(self, settings: ModelSettings) -> LoadedModel:
        with self._lock:
            try:
                loaded = self._loader(settings)
            except Exception as exc:  # MLflow raises many unrelated exception types
                logger.exception(
                    "Could not load model %s from %s",
                    settings.model_uri,
                    settings.mlflow_tracking_uri,
                )
                raise ModelUnavailableError(
                    f"Could not load the model {settings.model_uri}: {type(exc).__name__}",
                    hint=_unavailable_hint(settings),
                ) from exc
            self.current = loaded
            logger.info("Loaded model %s (version %s)", loaded.uri, loaded.version)
            return loaded

    def require(self, settings: ModelSettings) -> LoadedModel:
        current = self.current
        if current is None:
            raise ModelUnavailableError(
                "The sentiment model is not loaded", hint=_unavailable_hint(settings)
            )
        return current


def _unavailable_hint(settings: ModelSettings) -> str:
    return (
        f"Check that MLflow is running at {settings.mlflow_tracking_uri} and that "
        f"{settings.model_uri} exists (train and register it with python training/train.py), "
        "then call POST /api/model/reload."
    )


model_store = ModelStore()


def get_model_store() -> ModelStore:
    return model_store
