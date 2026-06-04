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
    assert "GET /app" in payload["endpoints"]


def test_app_serves_web_ui() -> None:
    client = TestClient(app)
    response = client.get("/app")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ResearchGraph-RAG" in response.text


def test_api_summary_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "ResearchGraph-RAG"
    assert payload["docs_url"] == "/docs"
    assert "POST /query" in payload["endpoints"]


def test_runtime_config_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_provider"] in {"openai", "huggingface", "ollama"}
    assert "embedding_provider" in payload
    assert "qdrant_collection" in payload
