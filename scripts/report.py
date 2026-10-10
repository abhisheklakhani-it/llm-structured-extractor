"""Summarise the repeated eval runs into a table (results/summary.json) and a chart (docs/accuracy.png).

    python scripts/report.py
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROMPTS = ["zero_shot", "improved"]
LABELS = {"zero_shot": "Zero-shot prompt", "improved": "Improved prompt"}
SPLITS = {"heldout": "Held-out set (10 postings, never used for tuning)", "dev": "Dev set (30 postings, prompt tuned on these)"}
METRICS = ["title", "company", "salary", "remote", "language", "skills F1", "all fields"]

# Reference palette: categorical slots 1-2, ink and chrome.
COLORS = {"zero_shot": "#2a78d6", "improved": "#eb6834"}
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def run_files(prompt: str, split: str) -> list[Path]:
    base = prompt if split == "dev" else f"{prompt}.{split}"
    return [RESULTS / f"{base}.json"] + sorted(RESULTS.glob(f"{base}.run*.json"))


def values(metrics: dict) -> dict[str, float]:
    fields = metrics["fields"]
    out = {f: fields[f]["accuracy"] for f in ["title", "company", "salary", "remote", "language"]}
    out["skills F1"] = fields["skills"]["f1"]
    out["skills precision"] = fields["skills"]["precision"]
    out["skills recall"] = fields["skills"]["recall"]
    out["all fields"] = metrics["all_scalar_fields_correct"]
    out["retries"] = metrics["retries"]
    out["median seconds"] = metrics["median_seconds"]
    return out


def summarise() -> dict:
    summary = {}
    for split in SPLITS:
        for prompt in PROMPTS:
            runs = [values(json.loads(p.read_text())) for p in run_files(prompt, split)]
            summary[f"{prompt}.{split}"] = {
                key: {"mean": statistics.mean(r[key] for r in runs), "min": min(r[key] for r in runs), "max": max(r[key] for r in runs)}
                for key in runs[0]
            } | {"runs": len(runs)}
    return summary


def chart(summary: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), facecolor=SURFACE)
    width = 0.36
    for ax, (split, title) in zip(axes, SPLITS.items()):
        ax.set_facecolor(SURFACE)
        for offset, prompt in zip((-width / 2 - 0.01, width / 2 + 0.01), PROMPTS):
            stats = summary[f"{prompt}.{split}"]
            means = [stats[m]["mean"] for m in METRICS]
            low = [stats[m]["mean"] - stats[m]["min"] for m in METRICS]
            high = [stats[m]["max"] - stats[m]["mean"] for m in METRICS]
            xs = [i + offset for i in range(len(METRICS))]
            ax.bar(xs, means, width, color=COLORS[prompt], label=LABELS[prompt], zorder=2)
            ax.errorbar(xs, means, yerr=[low, high], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
        # Label only the headline metric, so the chart is not covered in numbers.
        last = len(METRICS) - 1
        for offset, prompt in zip((-width / 2 - 0.01, width / 2 + 0.01), PROMPTS):
            mean = summary[f"{prompt}.{split}"]["all fields"]["mean"]
            ax.text(last + offset, mean + 0.04, f"{mean:.0%}", ha="center", va="bottom", color=INK, fontsize=9, fontweight="bold")
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
        ax.set_xticks(range(len(METRICS)), METRICS, color=INK_2)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.axvspan(last - 0.5, last + 0.5, color=GRID, alpha=0.35, zorder=0)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=INK_2)
    fig.suptitle("Per-field accuracy, mean of 3 runs (whiskers = min-max)  ·  gpt-oss:120b", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "accuracy.png", dpi=160, facecolor=SURFACE)


def main() -> None:
    summary = summarise()
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    chart(summary)
    keys = ["title", "company", "salary", "remote", "language", "skills precision", "skills recall", "skills F1", "all fields"]
    for split in SPLITS:
        print(f"\n{split}")
        print("| Field | " + " | ".join(LABELS[p] for p in PROMPTS) + " |")
        print("|---|---|---|")
        for key in keys:
            cells = []
            for prompt in PROMPTS:
                s = summary[f"{prompt}.{split}"][key]
                cells.append(f"{s['mean']:.2f} ({s['min']:.2f}–{s['max']:.2f})")
            print(f"| {key} | " + " | ".join(cells) + " |")
        for prompt in PROMPTS:
            s = summary[f"{prompt}.{split}"]
            print(f"{prompt}: retries {s['retries']['mean']:.1f}, median {s['median seconds']['mean']:.1f}s, runs {s['runs']}")


if __name__ == "__main__":
    main()
