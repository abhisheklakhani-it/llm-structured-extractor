"""Settings read from environment variables (and .env)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODELS = {"ollama": "gpt-oss:120b", "anthropic": "claude-opus-5-5"}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    ollama_host: str
    ollama_api_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        return cls(
            provider=provider,
            model=os.getenv("EXTRACT_MODEL") or DEFAULT_MODELS.get(provider, ""),
            ollama_host=os.getenv("OLLAMA_HOST", "https://ollama.com"),
            ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
        )
