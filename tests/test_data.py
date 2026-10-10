"""The gold sets must match the postings and pass the same schema the model must pass."""

from pathlib import Path

import pytest

from jobextract.evaluation import load_gold
from jobextract.schema import JobPosting

DATA = Path(__file__).resolve().parents[1] / "data"
SPLITS = [("postings", "gold.jsonl", 30), ("heldout", "gold_heldout.jsonl", 10)]


@pytest.mark.parametrize("folder, gold_file, size", SPLITS)
def test_every_posting_has_a_gold_label(folder, gold_file, size):
    golds = load_gold(DATA / gold_file)
    files = {p.stem for p in (DATA / folder).glob("*.txt")}
    assert len(golds) == size
    assert set(golds) == files


@pytest.mark.parametrize("folder, gold_file, size", SPLITS)
def test_gold_labels_are_valid(folder, gold_file, size):
    for pid, gold in load_gold(DATA / gold_file).items():
        JobPosting.model_validate(gold)


def test_dev_set_mixes_languages():
    languages = [g["language"] for g in load_gold(DATA / "gold.jsonl").values()]
    assert languages.count("de") >= 10 and languages.count("en") >= 10
