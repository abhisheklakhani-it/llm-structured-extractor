import pytest

from jobextract.config import Settings
from jobextract.llm import ConfigError, OllamaLLM, make_llm
from jobextract.schema import LLMExtraction


def settings(**overrides):
    base = dict(provider="ollama", model="gpt-oss:120b", ollama_host="https://ollama.com", ollama_api_key="key")
    return Settings(**{**base, **overrides})


def test_unknown_provider_is_rejected():
    with pytest.raises(ConfigError, match="Unknown LLM_PROVIDER"):
        make_llm(settings(provider="nope"))


def test_ollama_cloud_needs_a_key():
    with pytest.raises(ConfigError, match="OLLAMA_API_KEY"):
        make_llm(settings(ollama_api_key=""))


def test_local_ollama_needs_no_key():
    assert isinstance(make_llm(settings(ollama_host="http://localhost:11434", ollama_api_key="")), OllamaLLM)


def test_claude_needs_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        make_llm(settings(provider="anthropic"))


def test_ollama_sends_schema_in_prompt_and_format():
    llm = OllamaLLM(settings())
    sent = {}

    class Reply:
        class message:
            content = "{}"

    def fake_chat(**kwargs):
        sent.update(kwargs)
        return Reply

    llm.client.chat = fake_chat
    llm.complete("SYSTEM", "USER", LLMExtraction)
    assert sent["format"] == LLMExtraction.model_json_schema()
    assert sent["messages"][0]["content"].startswith("SYSTEM")
    assert '"confidence"' in sent["messages"][0]["content"]
    assert sent["options"]["temperature"] == 0
