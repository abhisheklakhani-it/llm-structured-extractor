"""The Pydantic schema every extraction must pass."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Salary(BaseModel):
    min: int | None = Field(description="Lowest amount, a whole number in the currency unit")
    max: int | None = Field(description="Highest amount, a whole number in the currency unit")
    currency: str = Field(description="ISO 4217 code, e.g. EUR")
    period: Literal["year", "month", "hour"]

    @field_validator("currency")
    @classmethod
    def iso_code(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError(f"currency must be a 3-letter ISO code, got {value!r}")
        return value

    @model_validator(mode="after")
    def check_range(self) -> "Salary":
        if self.min is None and self.max is None:
            raise ValueError("salary needs at least min or max; use null for the whole salary instead")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"salary min {self.min} is greater than max {self.max}")
        return self


class JobPosting(BaseModel):
    title: str = Field(description="Job title")
    company: str | None = Field(description="Hiring company, or null")
    skills: list[str] = Field(description="Technical skills and tools")
    salary: Salary | None = Field(description="Salary, or null if none is given")
    remote: Literal["remote", "hybrid", "onsite", "unspecified"]
    language: Literal["en", "de"] = Field(description="Language the posting is written in")

    @field_validator("title")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be empty")
        return value.strip()

    @field_validator("skills")
    @classmethod
    def dedupe_skills(cls, value: list[str]) -> list[str]:
        seen, out = set(), []
        for skill in (s.strip() for s in value):
            if skill and skill.lower() not in seen:
                seen.add(skill.lower())
                out.append(skill)
        return out
