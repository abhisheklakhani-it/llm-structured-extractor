"""Prompts for the extractor."""

ZERO_SHOT_SYSTEM = "Extract the job posting's details as JSON matching the given schema."


def user_message(posting: str) -> str:
    return f"<posting>\n{posting.strip()}\n</posting>"


def retry_message(posting: str, previous_reply: str, error: str) -> str:
    return (
        f"{user_message(posting)}\n\n"
        f"Your previous reply was rejected.\n<previous_reply>\n{previous_reply.strip()}\n</previous_reply>\n"
        f"<error>\n{error}\n</error>\n"
        "Return a corrected JSON object that fixes this error."
    )
