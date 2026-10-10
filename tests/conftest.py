import json

import pytest

VALID = {
    "title": "Data Engineer",
    "company": "Acme GmbH",
    "skills": ["Python", "SQL"],
    "salary": {"min": 60000, "max": 70000, "currency": "EUR", "period": "year"},
    "remote": "hybrid",
    "language": "en",
    "confidence": 0.9,
}


class FakeLLM:
    """Returns queued replies and records every call, so tests need no network."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, schema):
        self.calls.append((system, user))
        return self.replies.pop(0)


@pytest.fixture
def valid_json() -> str:
    return json.dumps(VALID)


@pytest.fixture
def fake_llm():
    return FakeLLM
