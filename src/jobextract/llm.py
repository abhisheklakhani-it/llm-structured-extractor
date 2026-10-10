"""LLM clients that return JSON constrained to a Pydantic schema.

Claude's structured outputs constrain decoding, so the reply always matches the schema's
shape. Ollama's `format` does the same on a local server, but Ollama Cloud ignored it in
testing (the model invented its own fields), so the Ollama client also writes the schema
into the system prompt. Pydantic validates every result either way, because a schema
cannot express every rule (for example, salary min <= max).
"""

import json
import os
from typing import Protocol

from pydantic import BaseModel

from .config import DEFAULT_MODELS, Settings


class ConfigError(RuntimeError):
    """Raised when the LLM provider is unknown or its credentials are missing."""


class StructuredLLM(Protocol):
    def complete(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Return the model's raw JSON text for the given schema."""


class OllamaLLM:
    """Ollama's native API (Ollama Cloud or a local server), with the schema in `format` and in the prompt."""

    def __init__(self, settings: Settings):
        from ollama import Client

        headers = {"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else {}
        if "ollama.com" in settings.ollama_host and not headers:
            raise ConfigError("Set OLLAMA_API_KEY to use Ollama Cloud (create one at ollama.com/settings/keys).")
        self.model = settings.model
        self.client = Client(host=settings.ollama_host, headers=headers, timeout=120)

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> str:
        json_schema = schema.model_json_schema()
        system = f"{system}\n\nReply with only a JSON object, no markdown, matching this JSON schema:\n{json.dumps(json_schema)}"
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=json_schema,
            options={"temperature": 0},
        )
        return response.message.content


class AnthropicLLM:
    """Claude with structured outputs via `messages.parse`."""

    def __init__(self, settings: Settings):
        import anthropic

        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise ConfigError("Set ANTHROPIC_API_KEY to use Claude.")
        self.model = settings.model
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> str:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return next(block.text for block in response.content if block.type == "text")


def make_llm(settings: Settings) -> StructuredLLM:
    if settings.provider == "ollama":
        return OllamaLLM(settings)
    if settings.provider == "anthropic":
        return AnthropicLLM(settings)
    raise ConfigError(f"Unknown LLM_PROVIDER {settings.provider!r}; use one of: {', '.join(DEFAULT_MODELS)}.")
