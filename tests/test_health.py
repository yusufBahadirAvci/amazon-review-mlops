import pytest

from app.db import database_is_up
from app.main import app


@pytest.fixture
def client(api_client):
    # api_client already swaps in an empty model store, so no MLflow is needed
    return api_client


def test_health_ok_when_database_is_up(client):
    app.dependency_overrides[database_is_up] = lambda: True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "up"


def test_health_503_when_database_is_down(client):
    app.dependency_overrides[database_is_up] = lambda: False

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "down"


def test_health_reports_unavailable_model(client):
    app.dependency_overrides[database_is_up] = lambda: True

    body = client.get("/health").json()

    assert (body["model"], body["model_version"]) == ("unavailable", None)


def test_health_reports_loaded_model(client, model_store, model_settings):
    app.dependency_overrides[database_is_up] = lambda: True
    model_store.load(model_settings)

    body = client.get("/health").json()

    assert body == {"status": "ok", "database": "up", "model": "loaded", "model_version": "1"}
