import json

import pytest

from conftest import VALID
from jobextract.extractor import ExtractionError, Extractor, clean_json


def test_clean_json_strips_markdown_fences():
    assert clean_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_clean_json_strips_text_around_the_object():
    assert clean_json('Here you go: {"a": {"b": 2}} Hope this helps!') == '{"a": {"b": 2}}'


def test_valid_reply_needs_one_attempt(fake_llm, valid_json):
    result = Extractor(fake_llm(valid_json)).extract("Data Engineer at Acme")
    assert result.attempts == 1
    assert result.errors == []
    assert result.confidence == 0.9
    assert result.posting.title == "Data Engineer"


def test_fenced_reply_is_accepted(fake_llm, valid_json):
    result = Extractor(fake_llm(f"```json\n{valid_json}\n```")).extract("posting")
    assert result.attempts == 1


def test_invalid_reply_is_retried_with_the_error(fake_llm, valid_json):
    bad = json.dumps({**VALID, "salary": {"min": 9, "max": 1, "currency": "EUR", "period": "year"}})
    llm = fake_llm(bad, valid_json)
    result = Extractor(llm).extract("posting")
    assert result.attempts == 2
    assert "greater than max" in result.errors[0]
    retry_prompt = llm.calls[1][1]
    assert "previous reply was rejected" in retry_prompt
    assert "greater than max" in retry_prompt


def test_broken_json_is_retried(fake_llm, valid_json):
    result = Extractor(fake_llm('{"title": "Data', valid_json)).extract("posting")
    assert result.attempts == 2


def test_gives_up_after_max_attempts(fake_llm):
    llm = fake_llm("nope", "still nope")
    with pytest.raises(ExtractionError) as info:
        Extractor(llm, max_attempts=2).extract("posting")
    assert len(info.value.errors) == 2
    assert len(llm.calls) == 2


def test_system_prompt_is_passed_through(fake_llm, valid_json):
    llm = fake_llm(valid_json)
    Extractor(llm, system_prompt="RULES").extract("posting text")
    system, user = llm.calls[0]
    assert system == "RULES"
    assert "<posting>\nposting text\n</posting>" in user
