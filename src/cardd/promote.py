"""Promotion decision logic: compare a challenger's metrics.json against the current champion's,
and (if it wins) publish the change. Separate from training entirely - this module never trains
anything; it only ever reads metrics.json files that `cardd-train`/`cardd-evaluate` already
produced, on Kaggle or wherever, and decides/records what happens next.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cardd.report import metrics_to_markdown

# Two separate marker pairs, one per document splice_marked_section() targets. Kept distinct
# (rather than one generic pair reused everywhere) so a marker search in one file can never
# accidentally match content meant for the other.
README_START_MARKER = "<!-- promote:champion-results:start -->"
README_END_MARKER = "<!-- promote:champion-results:end -->"
MODEL_CARD_START_MARKER = "<!-- promote:model-card-results:start -->"
MODEL_CARD_END_MARKER = "<!-- promote:model-card-results:end -->"


def decide_promotion(
    champion: dict,
    challenger: dict,
    split: str = "test",
    recall_tolerance: float = 0.0,
) -> dict:
    """Compare a challenger's metrics against the current champion's, on `split`.

    Promotion rule: recall must not regress by more than `recall_tolerance`. This project's own
    stated priority (README's Results Analysis) is recall over aggregate mAP - a missed detection
    is a worse outcome than a false positive for the reconditioning-assessment use case this feeds
    into - so recall is what gates the decision. mAP50-95 is included in the returned dict for
    visibility only; it never blocks a promotion on its own.

    Raises ValueError if `split` is missing from either metrics dict's "splits" - fail loudly
    rather than silently promoting or rejecting on malformed input, matching this repo's existing
    style (e.g. the category-contiguity assertion in data/coco.py).
    """
    for label, metrics in (("champion", champion), ("challenger", challenger)):
        if split not in metrics.get("splits", {}):
            available = list(metrics.get("splits", {}))
            raise ValueError(f"{label} metrics has no '{split}' split (found: {available})")

    champion_agg = champion["splits"][split]["aggregate"]
    challenger_agg = challenger["splits"][split]["aggregate"]

    champion_recall = champion_agg["recall"]
    challenger_recall = challenger_agg["recall"]
    delta_recall = challenger_recall - champion_recall
    promote = delta_recall >= -recall_tolerance

    reason = (
        f"challenger recall {challenger_recall:.4f} vs champion {champion_recall:.4f} "
        f"(delta {delta_recall:+.4f}, tolerance {recall_tolerance:.4f}) - "
        f"{'promoting' if promote else 'rejecting'}"
    )

    return {
        "promote": promote,
        "reason": reason,
        "split": split,
        "champion_recall": champion_recall,
        "challenger_recall": challenger_recall,
        "delta_recall": delta_recall,
        "champion_map50_95": champion_agg["map50_95"],
        "challenger_map50_95": challenger_agg["map50_95"],
    }


def build_champion_pointer(
    run_name: str,
    release_tag: str,
    rolling_release_tag: str,
    weights_url: str,
    git_sha: str | None,
    metrics_path: str = "models/champion_metrics.json",
) -> dict:
    """Builds models/champion.json's content - a small audit/pointer record, distinct from the
    metrics themselves (models/champion_metrics.json), recording which promotion produced the
    currently-live champion and where its permanent (versioned) and rolling release assets live.
    """
    return {
        "schema_version": 1,
        "run_name": run_name,
        "release_tag": release_tag,
        "rolling_release_tag": rolling_release_tag,
        "weights_url": weights_url,
        "git_sha": git_sha,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "metrics_path": metrics_path,
    }


def splice_marked_section(
    text: str,
    new_body: str,
    start_marker: str,
    end_marker: str,
) -> str:
    """Replace the content between start_marker and end_marker (both kept, content between them
    replaced) with new_body. Generic - used for both README.md and MODEL_CARD.md, with a
    different marker pair for each (see README_*/MODEL_CARD_* constants above).

    Raises ValueError if either marker is missing - a future doc restructure that accidentally
    drops or renames these markers should break loudly here, not silently leave that section
    stale forever.
    """
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"Document is missing the markers ({start_marker!r} / {end_marker!r})")
    if end_idx < start_idx:
        raise ValueError("End marker appears before start marker")

    content_start = start_idx + len(start_marker)
    return text[:content_start] + "\n" + new_body.strip() + "\n" + text[end_idx:]


def apply_promotion(
    challenger_metrics_path: Path,
    champion_json_path: Path,
    champion_metrics_path: Path,
    readme_path: Path,
    release_tag: str,
    rolling_release_tag: str,
    weights_url: str,
    git_sha: str | None = None,
    model_card_path: Path | None = None,
) -> dict:
    """The one function in this module that touches disk.

    Copies the challenger's metrics.json to become the new champion_metrics.json, writes
    champion.json via build_champion_pointer(), and rewrites README's (and, if given,
    MODEL_CARD.md's) marked section using report.py's metrics_to_markdown() +
    splice_marked_section(). Both docs get the identical rendering - only the marker pair differs
    - since it's the same underlying champion_metrics.json either way. `model_card_path` is
    optional (unlike readme_path) so callers/tests that don't care about the model card aren't
    forced to set one up; when given, it's held to the same fail-loudly-on-missing-markers
    standard as the README. Returns the pointer dict written.
    """
    with open(challenger_metrics_path) as f:
        challenger_metrics = json.load(f)

    champion_metrics_path = Path(champion_metrics_path)
    champion_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(champion_metrics_path, "w") as f:
        json.dump(challenger_metrics, f, indent=2)

    pointer = build_champion_pointer(
        run_name=challenger_metrics["run_name"],
        release_tag=release_tag,
        rolling_release_tag=rolling_release_tag,
        weights_url=weights_url,
        git_sha=git_sha,
        metrics_path=str(champion_metrics_path),
    )
    champion_json_path = Path(champion_json_path)
    champion_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(champion_json_path, "w") as f:
        json.dump(pointer, f, indent=2)

    new_body = metrics_to_markdown(champion_metrics_path)

    readme_path = Path(readme_path)
    readme_text = readme_path.read_text()
    readme_path.write_text(
        splice_marked_section(readme_text, new_body, README_START_MARKER, README_END_MARKER)
    )

    if model_card_path is not None:
        model_card_path = Path(model_card_path)
        model_card_text = model_card_path.read_text()
        model_card_path.write_text(
            splice_marked_section(
                model_card_text, new_body, MODEL_CARD_START_MARKER, MODEL_CARD_END_MARKER
            )
        )

    return pointer
