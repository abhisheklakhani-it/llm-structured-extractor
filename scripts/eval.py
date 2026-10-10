"""Run the extractor over every posting and score it against the gold set.

    python scripts/eval.py --prompt zero_shot
    python scripts/eval.py --prompt improved
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobextract.config import Settings  # noqa: E402
from jobextract.evaluation import errors, load_gold, score  # noqa: E402
from jobextract.extractor import ExtractionError, Extractor  # noqa: E402
from jobextract.llm import make_llm  # noqa: E402
from jobextract.prompts import PROMPTS  # noqa: E402


SPLITS = {"dev": ("postings", "gold.jsonl"), "heldout": ("heldout", "gold_heldout.jsonl")}


def run_one(extractor: Extractor, folder: str, pid: str) -> dict:
    text = (ROOT / "data" / folder / f"{pid}.txt").read_text(encoding="utf-8")
    start = time.perf_counter()
    try:
        result = extractor.extract(text)
        row = {"id": pid, **result.model_dump()}
    except ExtractionError as error:
        row = {"id": pid, "posting": None, "confidence": None, "attempts": extractor.max_attempts, "errors": error.errors}
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="zero_shot")
    parser.add_argument("--name", help="results file name (default: the prompt name)")
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev",
                        help="dev = the 30 postings the prompt was tuned on; heldout = 10 postings it never saw")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rescore", action="store_true", help="re-score saved predictions without calling the LLM")
    args = parser.parse_args()

    folder, gold_file = SPLITS[args.split]
    golds = load_gold(ROOT / "data" / gold_file)
    name = args.name or (args.prompt if args.split == "dev" else f"{args.prompt}.{args.split}")
    predictions_path = ROOT / "results" / f"{name}.predictions.jsonl"
    settings = Settings.from_env()

    if args.rescore:
        with open(predictions_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    else:
        extractor = Extractor(make_llm(settings), system_prompt=PROMPTS[args.prompt])
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(lambda pid: run_one(extractor, folder, pid), sorted(golds)))

    preds = {row["id"]: row["posting"] for row in rows}
    metrics = score(preds, golds)
    ok = [row for row in rows if row["posting"] is not None]
    metrics.update(
        prompt=args.prompt,
        split=args.split,
        provider=settings.provider,
        model=settings.model,
        retries=sum(row["attempts"] - 1 for row in rows),
        median_seconds=statistics.median(row["seconds"] for row in rows),
        mean_confidence=statistics.mean(row["confidence"] for row in ok) if ok else None,
        errors=errors(preds, golds),
    )

    (ROOT / "results").mkdir(exist_ok=True)
    if not args.rescore:
        with open(predictions_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{name} ({args.split}): {settings.provider}/{settings.model}, {metrics['n']} postings, {metrics['failed']} failed, {metrics['retries']} retries")
    for field, values in metrics["fields"].items():
        print(f"  {field:<9} " + "  ".join(f"{k} {v:.2f}" for k, v in values.items()))
    print(f"  all scalar fields correct: {metrics['all_scalar_fields_correct']:.2f}")


if __name__ == "__main__":
    main()
