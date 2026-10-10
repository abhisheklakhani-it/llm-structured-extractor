import pytest
from pydantic import ValidationError

from jobextract.schema import JobPosting, LLMExtraction, Salary
from conftest import VALID


def test_valid_posting_parses():
    posting = JobPosting.model_validate({k: v for k, v in VALID.items() if k != "confidence"})
    assert posting.salary.currency == "EUR"


def test_currency_is_upper_cased_and_must_be_three_letters():
    assert Salary(min=1, max=2, currency="eur", period="year").currency == "EUR"
    with pytest.raises(ValidationError):
        Salary(min=1, max=2, currency="€", period="year")


def test_salary_min_above_max_is_rejected():
    with pytest.raises(ValidationError, match="greater than max"):
        Salary(min=90000, max=60000, currency="EUR", period="year")


def test_salary_needs_an_amount():
    with pytest.raises(ValidationError):
        Salary(min=None, max=None, currency="EUR", period="year")


def test_open_ended_salary_is_allowed():
    assert Salary(min=None, max=85000, currency="EUR", period="year").max == 85000


def test_unknown_remote_value_is_rejected():
    with pytest.raises(ValidationError):
        JobPosting.model_validate({**VALID, "remote": "sometimes"})


def test_skills_are_deduplicated_case_insensitively():
    posting = JobPosting.model_validate({**VALID, "skills": ["Python", "python ", "SQL", ""]})
    assert posting.skills == ["Python", "SQL"]


def test_empty_title_is_rejected():
    with pytest.raises(ValidationError):
        JobPosting.model_validate({**VALID, "title": "  "})


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_must_be_between_0_and_1(confidence):
    with pytest.raises(ValidationError):
        LLMExtraction.model_validate({**VALID, "confidence": confidence})
