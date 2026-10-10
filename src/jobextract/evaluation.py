"""Score predictions against the gold set, field by field."""

import json
import re
import unicodedata
from pathlib import Path

SCALAR_FIELDS = ["title", "company", "salary", "remote", "language"]

# Different spellings of the same skill. Keys and values are already normalised.
SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "google cloud platform (gcp)": "gcp",
    "amazon web services": "aws",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "bash scripting": "bash",
    "bash-scripting": "bash",
    "shell scripting": "bash",
    "huggingface": "hugging face",
    "hugging face transformers": "transformers",
    "huggingface transformers": "transformers",
    "golang": "go",
    "js": "javascript",
    "ts": "typescript",
    "spark": "apache spark",
    "kafka": "apache kafka",
    "flink": "apache flink",
    "airflow": "apache airflow",
    "rest": "rest api",
    "rest apis": "rest api",
    "large language models": "llms",
    "llm": "llms",
    "sentence-transformers": "sentence transformers",
    "a/b tests": "a/b testing",
    "ci/cd pipelines": "ci/cd",
    "asr": "speech recognition",
    "automatic speech recognition": "speech recognition",
    "aws guardduty": "guardduty",
    "aws security hub": "security hub",
}

_DASHES = re.compile(r"[‐-―−]")


def norm(text: str | None) -> str | None:
    """Lower-case, NFKC-normalise and unify dashes and spaces, so formatting noise is not an error."""
    if text is None:
        return None
    text = _DASHES.sub("-", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip().lower()


def norm_skill(skill: str) -> str:
    skill = norm(skill)
    skill = re.sub(r"\s*\((optional|nice to have|a plus|bonus)\)$", "", skill)
    return SKILL_ALIASES.get(skill, skill)


def field_correct(field: str, pred: dict | None, gold: dict) -> bool:
    if pred is None:  # the extraction failed
        return False
    if field in ("title", "company"):
        return norm(pred[field]) == norm(gold[field])
    return pred[field] == gold[field]


def score(preds: dict[str, dict | None], golds: dict[str, dict]) -> dict:
    """Accuracy for each scalar field, micro precision/recall/F1 for skills."""
    ids = sorted(golds)
    result = {"n": len(ids), "failed": sum(preds.get(i) is None for i in ids), "fields": {}}
    for field in SCALAR_FIELDS:
        correct = sum(field_correct(field, preds.get(i), golds[i]) for i in ids)
        result["fields"][field] = {"accuracy": correct / len(ids)}

    tp = n_pred = n_gold = 0
    for i in ids:
        gold_skills = {norm_skill(s) for s in golds[i]["skills"]}
        pred_skills = {norm_skill(s) for s in (preds.get(i) or {}).get("skills", [])}
        tp += len(gold_skills & pred_skills)
        n_pred += len(pred_skills)
        n_gold += len(gold_skills)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["fields"]["skills"] = {"precision": precision, "recall": recall, "f1": f1}

    all_correct = sum(all(field_correct(f, preds.get(i), golds[i]) for f in SCALAR_FIELDS) for i in ids)
    result["all_scalar_fields_correct"] = all_correct / len(ids)
    return result


def errors(preds: dict[str, dict | None], golds: dict[str, dict]) -> list[dict]:
    """Every wrong scalar field, for reading what went wrong."""
    out = []
    for i in sorted(golds):
        for field in SCALAR_FIELDS:
            if not field_correct(field, preds.get(i), golds[i]):
                predicted = None if preds.get(i) is None else preds[i][field]
                out.append({"id": i, "field": field, "gold": golds[i][field], "predicted": predicted})
    return out


def load_gold(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row.pop("id"): row for row in rows}
