import pytest
from fastapi.testclient import TestClient

from conftest import FakeLLM
from jobextract import api
from jobextract.llm import ConfigError

POSTING = "Data Engineer at Acme GmbH, Python and SQL, 60-70k EUR, hybrid."


@pytest.fixture
def client():
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def use_llm(llm):
    api.app.dependency_overrides[api.llm_factory] = lambda: (lambda: llm)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_extract_returns_validated_json(client, valid_json):
    use_llm(FakeLLM(valid_json))
    body = client.post("/extract", json={"text": POSTING}).json()
    assert body["posting"]["title"] == "Data Engineer"
    assert body["confidence"] == 0.9
    assert body["attempts"] == 1


def test_too_short_text_is_a_422(client):
    assert client.post("/extract", json={"text": "hi"}).status_code == 422


def test_model_that_never_returns_valid_json_is_a_502(client):
    use_llm(FakeLLM("bad", "bad", "bad"))
    response = client.post("/extract", json={"text": POSTING})
    assert response.status_code == 502
    assert len(response.json()["detail"]["errors"]) == 3


def test_missing_credentials_is_a_503(client):
    def broken():
        raise ConfigError("Set OLLAMA_API_KEY")

    api.app.dependency_overrides[api.llm_factory] = lambda: broken
    assert client.post("/extract", json={"text": POSTING}).status_code == 503
