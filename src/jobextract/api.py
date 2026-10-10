"""FastAPI service: POST a job posting, get validated JSON back."""

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .extractor import ExtractionError, Extractor
from .llm import ConfigError, StructuredLLM, make_llm
from .prompts import PROMPTS
from .schema import Extraction

app = FastAPI(title="LLM Structured Extractor", version="1.0.0")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20000, description="The job posting, as plain text")
    prompt: Literal["improved", "zero_shot"] = "improved"


@lru_cache
def get_llm() -> StructuredLLM:
    return make_llm(Settings.from_env())


def llm_factory():
    """Dependency that returns a callable, so a missing key becomes a 503 instead of a crash at startup."""
    return get_llm


@app.get("/health")
def health() -> dict:
    settings = Settings.from_env()
    return {"status": "ok", "provider": settings.provider, "model": settings.model}


@app.post("/extract", response_model=Extraction)
def extract(request: ExtractRequest, llm_factory=Depends(llm_factory)) -> Extraction:
    try:
        llm = llm_factory()
    except ConfigError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        return Extractor(llm, system_prompt=PROMPTS[request.prompt]).extract(request.text)
    except ExtractionError as error:
        raise HTTPException(status_code=502, detail={"message": "the model never returned valid JSON", "errors": error.errors}) from error
