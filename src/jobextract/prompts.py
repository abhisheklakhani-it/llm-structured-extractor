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


# Written from the zero-shot errors on the 30 dev postings (see README). The worked example
# is invented and is not one of the dev or held-out postings.
IMPROVED_SYSTEM = """You extract structured data from job postings written in English or German. The posting may be messy: scraped page text, a recruiter email, a social media post. Follow these rules exactly.

title
- The job title only. Remove gender tags such as (m/w/d), (w/m/d), (f/m/x), (m/f/d), (all genders). Gendered word endings such as "Entwickler:in" stay as written.
- Remove contract details (duration, "6 Monate", "extendable", "full-time"), the location and phrases about the team or product introduced by "for our" / "für unsere".
- Keep seniority and specialisation that belong to the title, such as "Senior", "– NLP", "für Dokumentenverarbeitung", "(Python)".

company
- The company that will employ the person, with its legal suffix as written (GmbH, AG, Ltd).
- If a recruiter or agency writes on behalf of an unnamed client, company is null. Never use the agency's name.

skills
- Technical skills only: programming languages, frameworks, libraries, tools, cloud platforms and services, and technical methods named in the duties or requirements (for example RAG, A/B testing, time series, object detection, LLM evaluation).
- One skill per entry. Split groups: "AWS (Lambda, S3)" becomes "AWS", "AWS Lambda", "S3". "Terraform or Pulumi" becomes both.
- Use the common English name, short and without qualifiers: "Data Modelling" not "Datenmodellierung", "Bash" not "Bash scripting", "Networking" not "network fundamentals". Drop "(optional)", "a plus".
- Do not include spoken languages, soft skills, degrees, certifications, regulations or years of experience.

salary
- null if the posting gives no amount ("competitive", "attractive", a collective agreement like TV-L without an amount).
- min and max are whole numbers in the currency unit: "65k" = 65000, "4.500 €" = 4500 (German notation uses a dot for thousands).
- "up to X" / "bis zu X" gives min null; "from X" / "ab X" gives max null; a single amount gives min = max.
- currency is the ISO code (EUR, USD, GBP, CHF); period is year, month or hour.

remote
- remote: fully remote, including "office optional", "remote-first", "office or fully remote, your choice", and remote with rare meetups.
- hybrid: a mix of office and home days, "home office possible", "mostly remote but on-site every two weeks".
- onsite: work at the office or site only.
- unspecified: the posting says nothing about where the work happens.

language
- The language the posting is written in (en or de), even if it contains words from the other language.

confidence
- From 0.0 to 1.0: how sure you are that every field is correct. Lower it when a field was ambiguous or had to be inferred.

Example
<posting>
Wir suchen eine:n Data Engineer (m/w/d) für unser Analytics-Team in Kassel. Tools: Python, AWS (Glue, Athena), dbt. Gehalt ab 58.000 € jährlich. Arbeite im Büro oder komplett remote, wie du magst. Gesendet von Jana, Recruiterin bei TopTalent Partners, für einen Kunden aus der Energiebranche.
</posting>
Answer:
{"title": "Data Engineer", "company": null, "skills": ["Python", "AWS", "AWS Glue", "Athena", "dbt"], "salary": {"min": 58000, "max": null, "currency": "EUR", "period": "year"}, "remote": "remote", "language": "de", "confidence": 0.9}"""


PROMPTS = {"zero_shot": ZERO_SHOT_SYSTEM, "improved": IMPROVED_SYSTEM}
