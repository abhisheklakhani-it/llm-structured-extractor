# Labelling guide

`gold.jsonl` holds one hand-labelled answer per posting in `postings/`. The 30 postings are
fictional (invented companies, no real people) but copy the formats you meet on LinkedIn,
StepStone, company career pages and recruiter emails: bullet lists, scraped page junk,
emoji, German salary notation, salaries hidden in the middle of a paragraph.

These rules decide every label, so the gold set is consistent and the evaluation is fair.

| Field | Rule |
|---|---|
| `title` | The job title as written, without gender tags like `(m/w/d)`, `(f/m/x)` or `(all genders)` and without location or seniority prefixes added by the job board. Keep seniority that is part of the title (`Senior Data Scientist`). |
| `company` | The company that will employ the person. A recruiter writing for an unnamed client gives `null`. Legal suffixes (`GmbH`, `AG`, `Ltd`) are kept as written. |
| `skills` | Technical skills, tools, programming languages, frameworks, platforms and methods named as required or nice to have. Not soft skills ("team player"), not spoken languages ("fluent German"), not degrees. Written in their common form (`PyTorch`, `Kubernetes`, `SQL`). |
| `salary` | `null` when no amount is given. Otherwise `min`, `max` (whole numbers in the currency unit, so `65k` = `65000` and `4.500 €` = `4500`), `currency` (ISO code: `EUR`, `USD`, `GBP`, `CHF`) and `period` (`year`, `month`, `hour`). "Up to 90k" gives `min: null, max: 90000`; a single figure gives `min = max`. |
| `remote` | `remote` (fully remote, maybe with rare on-site meetups), `hybrid` (some days in the office, or "home office possible"), `onsite` (office only), or `unspecified` when the posting says nothing. |
| `language` | The language the posting is written in: `en` or `de`. A German posting with English buzzwords is still `de`. |

Skills are scored as a set (precision and recall after lower-casing and an alias table), so
the order does not matter and `scikit-learn` matches `sklearn`.
