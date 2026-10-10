"""Turn a job posting into a validated JobPosting, retrying when the model's reply is invalid."""

import json
import re

from pydantic import ValidationError

from .llm import StructuredLLM
from .prompts import ZERO_SHOT_SYSTEM, retry_message, user_message
from .schema import Extraction, JobPosting, LLMExtraction

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class ExtractionError(RuntimeError):
    """Raised when every attempt returned invalid output."""

    def __init__(self, errors: list[str]):
        super().__init__(f"extraction failed after {len(errors)} attempts: {errors[-1]}")
        self.errors = errors


def clean_json(raw: str) -> str:
    """Strip markdown fences and any text around the outermost JSON object."""
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def describe(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in error.errors())
    return str(error)


class Extractor:
    def __init__(self, llm: StructuredLLM, system_prompt: str = ZERO_SHOT_SYSTEM, max_attempts: int = 3):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_attempts = max_attempts

    def extract(self, posting: str) -> Extraction:
        message = user_message(posting)
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            raw = self.llm.complete(self.system_prompt, message, LLMExtraction)
            try:
                result = LLMExtraction.model_validate(json.loads(clean_json(raw)))
            except (json.JSONDecodeError, ValidationError) as error:
                errors.append(describe(error))
                # Show the model its own reply and what was wrong with it, then ask again.
                message = retry_message(posting, raw, errors[-1])
                continue
            fields = result.model_dump(exclude={"confidence"})
            return Extraction(
                posting=JobPosting.model_validate(fields),
                confidence=result.confidence,
                attempts=attempt,
                errors=errors,
            )
        raise ExtractionError(errors)
