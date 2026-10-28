import pytest
from fastapi.testclient import TestClient

from app.db import database_is_up
from app.main import app


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_ok_when_database_is_up(client):
    app.dependency_overrides[database_is_up] = lambda: True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "up"}


def test_health_503_when_database_is_down(client):
    app.dependency_overrides[database_is_up] = lambda: False

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "down"}
