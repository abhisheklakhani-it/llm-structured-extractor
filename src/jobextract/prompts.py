"""Prompts for the extractor."""

ZERO_SHOT_SYSTEM = "Extract the job posting's details as JSON matching the given schema."


def user_message(posting: str) -> str:
    return f"<posting>\n{posting.strip()}\n</posting>"
