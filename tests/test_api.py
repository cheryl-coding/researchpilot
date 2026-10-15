from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from researchpilot.api.main import app


@pytest.fixture
def client():
    service = Mock()
    service.list_documents.return_value = []
    service.ask.return_value = {
        "answer": "A fact [1]",
        "sources": [{"id": 1, "document": "test.pdf", "page": 1, "text": "A fact"}],
        "attempts": 1,
        "trace": ["Retrieve", "Generate"],
    }
    app.state.service = service
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat(client):
    result = client.post("/chat", json={"question": "Question?"})
    assert result.status_code == 200
    assert result.json()["sources"][0]["page"] == 1


def test_invalid_mode(client):
    assert (
        client.post("/chat", json={"question": "Hi", "retrieval_mode": "invalid"}).status_code
        == 422
    )


def test_invalid_pdf(client):
    result = client.post(
        "/documents/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}
    )
    assert result.status_code == 400
