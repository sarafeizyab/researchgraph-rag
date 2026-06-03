import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "timestamp" in payload


def test_root_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "ResearchGraph-RAG"
    assert payload["docs_url"] == "/docs"
    assert "POST /query" in payload["endpoints"]
