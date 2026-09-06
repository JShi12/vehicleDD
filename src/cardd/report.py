"""Render metrics.json file(s) into the same markdown table format used in README.md - a manual
helper (not CI-automated) so a real Kaggle run's metrics.json can replace hand-transcription."""
from __future__ import annotations

import json
from pathlib import Path


def _aggregate_row(split_name: str, split: dict) -> str:
    agg = split["aggregate"]
    label = f"**{split_name}**" if split_name == "test" else split_name
    return (
        f"| {label} | {split['images']} | {split['instances']} | "
        f"{agg['precision']:.3f} | {agg['recall']:.3f} | "
        f"{agg['map50']:.3f} | {agg['map50_95']:.3f} |"
    )


def _per_class_table(per_class: list[dict]) -> str:
    lines = [
        "| class | precision | recall | F1 | AP50 | AP50-95 |",
        "|---|---|---|---|---|---|",
    ]
    for c in per_class:
        lines.append(
            f"| {c['class_name']} | {c['precision']:.3f} | {c['recall']:.3f} | "
            f"{c['f1']:.3f} | {c['ap50']:.3f} | {c['ap50_95']:.3f} |"
        )
    return "\n".join(lines)


def metrics_to_markdown(metrics_path: Path) -> str:
    with open(metrics_path) as f:
        metrics = json.load(f)

    lines = [f"## Results — `{metrics['run_name']}`", ""]
    if metrics.get("config_path"):
        lines.append(f"Config: `{metrics['config_path']}`  ")
    if metrics.get("train_seconds"):
        lines.append(f"Train time: {metrics['train_seconds'] / 3600:.3f} hours  ")
    lines += ["", "| split | images | instances | precision | recall | mAP50 | mAP50-95 |",
              "|---|---|---|---|---|---|---|"]
    for split_name in ("val", "test"):
        if split_name in metrics["splits"]:
            lines.append(_aggregate_row(split_name, metrics["splits"][split_name]))
    lines.append("")

    if "test" in metrics["splits"]:
        per_class_table = _per_class_table(metrics["splits"]["test"]["per_class"])
        lines += ["**Per-class (test set):**", "", per_class_table, ""]

    return "\n".join(lines)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics_json", type=Path, nargs="+")
    args = p.parse_args()
    for path in args.metrics_json:
        print(metrics_to_markdown(path))
        print()


if __name__ == "__main__":
    main()
