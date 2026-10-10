# Labelling guide

`gold.jsonl` holds one hand-labelled answer per posting in `postings/`. The 30 postings are
fictional (invented companies, no real people) but copy the formats you meet on LinkedIn,
StepStone, company career pages and recruiter emails: bullet lists, scraped page junk,
emoji, German salary notation, salaries hidden in the middle of a paragraph.

These rules decide every label, so the gold set is consistent and the evaluation is fair.

| Field | Rule |
|---|---|
| `title` | The job title as written, without gender tags like `(m/w/d)`, `(f/m/x)` or `(all genders)`, without contract details (`6 Monate`, `6 months, extendable`), without the location and without phrases about the team or product introduced by "for our" / "für unsere". Keep seniority, the kind of position and the specialisation (`Senior`, `Werkstudent`, `Praktikum`, `– NLP`, `für Dokumentenverarbeitung`, `(Python)`). Gendered word endings such as `Entwickler:in` stay. |
| `company` | The company that will employ the person. A recruiter writing for an unnamed client gives `null`. Legal suffixes (`GmbH`, `AG`, `Ltd`) are kept as written. |
| `skills` | Technical skills, tools, programming languages, frameworks, platforms and technical methods named in the duties or requirements, required or nice to have (`RAG`, `A/B Testing`, `Time Series`, `LLM Evaluation`). A named group and its members are all listed: `AWS (Lambda, S3)` gives `AWS`, `AWS Lambda`, `S3`; "vector databases (Qdrant, Pinecone)" gives `Vector Databases`, `Qdrant`, `Pinecone`. Not soft skills ("team player"), not spoken languages ("fluent German"), not degrees, certifications or regulations (`OSCP`, `IFRS 9`, `GDPR`). Written in their common English form (`PyTorch`, `Apache Kafka`, `Data Modelling`, `Bash`). |
| `salary` | `null` when no amount is given. Otherwise `min`, `max` (whole numbers in the currency unit, so `65k` = `65000` and `4.500 €` = `4500`), `currency` (ISO code: `EUR`, `USD`, `GBP`, `CHF`) and `period` (`year`, `month`, `hour`). "Up to 90k" gives `min: null, max: 90000`; a single figure gives `min = max`. |
| `remote` | `remote` (fully remote, maybe with rare on-site meetups), `hybrid` (some days in the office, or "home office possible"), `onsite` (office only), or `unspecified` when the posting says nothing. |
| `language` | The language the posting is written in: `en` or `de`. A German posting with English buzzwords is still `de`. |

Skills are scored as a set (precision and recall after lower-casing and an alias table), so
the order does not matter and `scikit-learn` matches `sklearn`.

## Label review (11 Oct 2026)

All 40 labels were re-read against this guide after the first evaluation runs. The guide
was updated to write down title and skills rules the labels already followed, and six
dev-set skill lists that broke the guide were fixed:

| Posting | Change | Rule |
|---|---|---|
| 004 | added `Vector Databases`, `LLM Evaluation` | named group and its members; methods named in the duties |
| 008 | `Kafka` → `Apache Kafka` | common form, as in every other posting |
| 017 | added `Experimentation` | method named in the requirements (as in 018) |
| 019 | added `Infrastructure as Code` | method named in the duties (as in h09) |
| 023 | added `Infrastructure as Code` | named group and its members |
| 025 | added `LambdaMART` | algorithm named in the duties |

The review happened after seeing model predictions, so it could favour the model on the
dev set. The 10 held-out labels needed no change, so the held-out scores are unaffected.
