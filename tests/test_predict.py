import mlflow
import mlflow.pyfunc
import mlflow.tracking
import pytest
import requests

from app.config import ModelSettings
from app.ml_model import ModelUnavailableError, load_at_startup, load_from_mlflow

REVIEWS = [
    "I love it, great quality.",
    "It is okay, nothing special.",
    "It broke after two days.",
]


@pytest.fixture
def loaded(model_store, model_settings):
    model_store.load(model_settings)
    return model_store


def test_predict(api_client, loaded, model_loader):
    response = api_client.post("/api/predict", json={"texts": REVIEWS})

    assert response.status_code == 200
    assert response.json() == {
        "predictions": ["positive", "neutral", "negative"],
        "model_version": "1",
    }
    # Contract: the model gets a DataFrame with only a "text" column, texts unchanged
    (frame,) = model_loader.model.inputs
    assert list(frame.columns) == ["text"]
    assert frame["text"].tolist() == REVIEWS


def test_predictions_are_plain_strings(api_client, loaded):
    # The fake returns a numpy array like the real model; JSON needs plain strings
    body = api_client.post("/api/predict", json={"texts": ["great"]}).json()

    assert body["predictions"] == ["positive"]
    assert type(body["predictions"][0]) is str


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"texts": []}, "texts"),
        ({"texts": ["fine"] * 101}, "texts"),
        ({"texts": ["fine", "   "]}, "texts.1"),
        ({"texts": [""]}, "texts.0"),
        ({"texts": ["x" * 5001]}, "texts.0"),
        ({"texts": "not a list"}, "texts"),
        ({"text": ["wrong key"]}, "texts"),
        ({"texts": [123]}, "texts.0"),
    ],
)
def test_invalid_input_returns_400(api_client, loaded, body, field):
    response = api_client.post("/api/predict", json=body)

    assert response.status_code == 400
    assert set(response.json()) == {"error", "hint"}
    assert response.json()["error"].startswith(field)


def test_limits_are_inclusive(api_client, loaded):
    response = api_client.post("/api/predict", json={"texts": ["x" * 5000] * 100})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 100


def test_predict_without_model_returns_503(api_client):
    response = api_client.post("/api/predict", json={"texts": REVIEWS})

    assert response.status_code == 503
    assert "not loaded" in response.json()["error"]
    assert "MLflow" in response.json()["hint"]
    assert "production" in response.json()["hint"]


def test_reload_loads_new_version(api_client, loaded):
    response = api_client.post("/api/model/reload")

    assert response.status_code == 200
    assert response.json() == {"model_version": "2", "previous_version": "1"}
    assert api_client.post("/api/predict", json={"texts": REVIEWS}).json()["model_version"] == "2"


def test_reload_makes_unavailable_model_available(api_client):
    assert api_client.post("/api/predict", json={"texts": REVIEWS}).status_code == 503

    response = api_client.post("/api/model/reload")

    assert response.json() == {"model_version": "1", "previous_version": None}
    assert api_client.post("/api/predict", json={"texts": REVIEWS}).status_code == 200


def test_failed_reload_keeps_serving_previous_model(api_client, loaded, model_loader):
    model_loader.fail = True

    response = api_client.post("/api/model/reload")

    assert response.status_code == 503
    assert "MLflow is not reachable" in response.json()["error"]
    prediction = api_client.post("/api/predict", json={"texts": REVIEWS})
    assert (prediction.status_code, prediction.json()["model_version"]) == (200, "1")


def test_failed_load_raises_and_leaves_store_empty(model_store, model_loader, model_settings):
    model_loader.fail = True

    with pytest.raises(ModelUnavailableError):
        model_store.load(model_settings)

    assert model_store.current is None


def test_startup_waits_for_mlflow_to_come_up(model_store, model_loader):
    model_loader.fail_times = 2
    sleeps = []
    settings = ModelSettings(_env_file=None, model_startup_wait_seconds=60)

    loaded = load_at_startup(model_store, settings, sleep=sleeps.append)

    assert loaded is not None
    assert model_loader.attempts == 3
    assert sleeps == [5, 5]


def test_startup_gives_up_after_the_wait(model_store, model_loader):
    model_loader.fail = True
    sleeps = []
    settings = ModelSettings(_env_file=None, model_startup_wait_seconds=12)

    assert load_at_startup(model_store, settings, sleep=sleeps.append) is None
    assert model_loader.attempts == 3
    assert sleeps == [5, 5]
    assert model_store.current is None


def test_startup_does_not_retry_a_missing_model(model_store, model_loader):
    model_loader.fail = True
    model_loader.error = LookupError("Registered model alias production not found")
    sleeps = []
    settings = ModelSettings(_env_file=None, model_startup_wait_seconds=60)

    assert load_at_startup(model_store, settings, sleep=sleeps.append) is None
    assert (model_loader.attempts, sleeps) == (1, [])


def test_startup_tries_once_by_default(model_store, model_loader, model_settings):
    model_loader.fail = True
    sleeps = []

    assert load_at_startup(model_store, model_settings, sleep=sleeps.append) is None
    assert (model_loader.attempts, sleeps) == (1, [])


class FakeVersion:
    version = "4"


def test_load_from_mlflow_pins_the_alias_to_a_version(monkeypatch):
    loaded_uris = []
    aliases = []

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            aliases.append((name, alias))
            return FakeVersion()

    monkeypatch.setattr(requests, "get", lambda url, timeout: None)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: loaded_uris.append(uri) or "m")

    loaded = load_from_mlflow(ModelSettings(_env_file=None))

    assert aliases == [("sentiment-model", "production")]
    assert loaded_uris == ["models:/sentiment-model/4"]
    assert (loaded.version, loaded.model) == ("4", "m")


def test_unreachable_mlflow_fails_fast(monkeypatch):
    checked = []

    def refuse(url, timeout):
        checked.append((url, timeout))
        raise requests.ConnectionError("connection refused")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("MLflow must not be called when the server is down")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", must_not_be_called)
    settings = ModelSettings(_env_file=None, mlflow_tracking_uri="http://localhost:5999/")

    with pytest.raises(ConnectionError, match="not reachable"):
        load_from_mlflow(settings)

    assert checked == [("http://localhost:5999/health", 5)]
