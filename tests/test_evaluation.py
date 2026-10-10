import pytest

from jobextract.evaluation import errors, norm, norm_skill, score

GOLD = {
    "1": {"title": "Data Engineer", "company": None, "skills": ["Python", "scikit-learn", "AWS"],
          "salary": None, "remote": "remote", "language": "en"},
    "2": {"title": "IT-Systemadministrator", "company": "Byteberg GmbH", "skills": ["Bash"],
          "salary": {"min": 1, "max": 2, "currency": "EUR", "period": "month"}, "remote": "onsite", "language": "de"},
}


def test_norm_ignores_case_spaces_and_dash_variants():
    assert norm("IT‑Systemadministrator  Linux") == norm("it-systemadministrator linux")
    assert norm(None) is None


@pytest.mark.parametrize("raw, expected", [
    ("sklearn", "scikit-learn"),
    ("Bash scripting", "bash"),
    ("Terraform (optional)", "terraform"),
    ("Google Cloud Platform", "gcp"),
    ("PyTorch", "pytorch"),
])
def test_skill_aliases(raw, expected):
    assert norm_skill(raw) == expected


def test_perfect_predictions_score_one():
    result = score(GOLD, GOLD)
    assert all(v["accuracy"] == 1 for k, v in result["fields"].items() if k != "skills")
    assert result["fields"]["skills"]["f1"] == 1
    assert result["all_scalar_fields_correct"] == 1


def test_failed_extraction_counts_as_wrong():
    result = score({"1": GOLD["1"], "2": None}, GOLD)
    assert result["failed"] == 1
    assert result["fields"]["title"]["accuracy"] == 0.5
    assert result["fields"]["skills"]["recall"] == 3 / 4


def test_skill_precision_and_recall():
    pred = {"1": {**GOLD["1"], "skills": ["python", "sklearn", "Docker"]}, "2": GOLD["2"]}
    skills = score(pred, GOLD)["fields"]["skills"]
    assert skills["precision"] == 3 / 4  # Docker is extra
    assert skills["recall"] == 3 / 4  # AWS is missing


def test_errors_lists_each_wrong_field():
    pred = {"1": {**GOLD["1"], "remote": "hybrid"}, "2": GOLD["2"]}
    assert errors(pred, GOLD) == [{"id": "1", "field": "remote", "gold": "remote", "predicted": "hybrid"}]
