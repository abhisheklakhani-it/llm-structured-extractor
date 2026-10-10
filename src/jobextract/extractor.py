"""Turn a job posting into a validated JobPosting."""

from .llm import StructuredLLM
from .prompts import ZERO_SHOT_SYSTEM, user_message
from .schema import JobPosting


class Extractor:
    def __init__(self, llm: StructuredLLM, system_prompt: str = ZERO_SHOT_SYSTEM):
        self.llm = llm
        self.system_prompt = system_prompt

    def extract(self, posting: str) -> JobPosting:
        raw = self.llm.complete(self.system_prompt, user_message(posting), JobPosting)
        return JobPosting.model_validate_json(raw)
