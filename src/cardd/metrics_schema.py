"""metrics.json schema: aggregate + per-class detection metrics for a val and/or test split.

Field provenance (Ultralytics `DetMetrics`-shaped `.box` accessor, as returned by both
`model.train()`'s validator and a standalone `model.val()` call):
  aggregate.precision  <- box.mp
  aggregate.recall     <- box.mr
  aggregate.map50      <- box.map50
  aggregate.map50_95   <- box.map
  per_class[i]         <- box.ap_class_index[i] (class_id), box.class_result(i) -> (p, r, ap50, ap),
                          box.f1[i]; class_name resolved from the checkpoint's own `model.names`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cardd.util import git_sha

SCHEMA_VERSION = 1


def build_split_metrics(box_metrics: Any, class_names: dict[int, str]) -> dict:
    """`box_metrics` is anything exposing the same shape as Ultralytics' `.box` accessor:
    .mp, .mr, .map50, .map, .ap_class_index, .class_result(i), .f1[i]. Kept duck-typed
    (not type-hinted to the real Ultralytics class) so this is testable with a plain fake object.
    """
    per_class = []
    for i, class_id in enumerate(box_metrics.ap_class_index):
        class_id = int(class_id)
        precision, recall, ap50, ap = box_metrics.class_result(i)
        per_class.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(box_metrics.f1[i]),
            "ap50": float(ap50),
            "ap50_95": float(ap),
        })
    return {
        "aggregate": {
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
        },
        "per_class": per_class,
    }


def build_metrics_dict(
    run_name: str,
    config_path: Path,
    weights_path: Path,
    class_names: dict[int, str],
    val_results: Any = None,
    test_results: Any = None,
    val_counts: dict | None = None,
    test_counts: dict | None = None,
    train_seconds: float | None = None,
    mlflow_run_id: str | None = None,
) -> dict:
    if val_results is None and test_results is None:
        raise ValueError("At least one of val_results/test_results must be provided")

    splits = {}
    if val_results is not None:
        val_metrics = build_split_metrics(val_results.box, class_names)
        splits["val"] = {**(val_counts or {}), **val_metrics}
    if test_results is not None:
        test_metrics = build_split_metrics(test_results.box, class_names)
        splits["test"] = {**(test_counts or {}), **test_metrics}

    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "config_path": str(config_path),
        "git_sha": git_sha(),
        "mlflow_run_id": mlflow_run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_seconds": train_seconds,
        "weights_path": str(weights_path),
        "class_names": [class_names[i] for i in sorted(class_names)],
        "splits": splits,
    }


def write_metrics_json(metrics: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return out_path


REQUIRED_SPLIT_KEYS = {"aggregate", "per_class"}
REQUIRED_AGGREGATE_KEYS = {"precision", "recall", "map50", "map50_95"}
REQUIRED_PER_CLASS_KEYS = {"class_id", "class_name", "precision", "recall", "f1", "ap50", "ap50_95"}


def validate_metrics_schema(metrics: dict) -> None:
    """Raises AssertionError on the first schema violation found. Used by tests and can be used
    as a post-write sanity check in the training CLI."""
    assert metrics.get("schema_version") == SCHEMA_VERSION
    assert "splits" in metrics and metrics["splits"], "metrics.json must have at least one split"
    for split_name, split in metrics["splits"].items():
        missing = REQUIRED_SPLIT_KEYS - split.keys()
        assert not missing, f"split '{split_name}' missing keys: {missing}"
        missing = REQUIRED_AGGREGATE_KEYS - split["aggregate"].keys()
        assert not missing, f"split '{split_name}'.aggregate missing keys: {missing}"
        for pc in split["per_class"]:
            missing = REQUIRED_PER_CLASS_KEYS - pc.keys()
            assert not missing, f"split '{split_name}'.per_class entry missing keys: {missing}"
