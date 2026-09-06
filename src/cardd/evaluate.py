"""Standalone evaluation: run a trained checkpoint's `.val()` against a chosen split.

Kept separate from training-time validation on purpose - training-time val (watched during
training, used for early stopping) is a different thing from the held-out **test** evaluation this
repo treats as the headline number. `run_training` in train.py calls this explicitly as a second,
separate `.val()` call after training finishes, exactly like the original notebooks did.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def run_evaluation(
    weights: Path,
    data_yaml: Path,
    split: str = "test",
    project: str | None = None,
    name: str | None = None,
) -> tuple[Any, dict[int, str]]:
    # A *relative* project is silently redirected by Ultralytics to a subdirectory of a global,
    # machine-level default runs directory rather than one resolved against cwd (see train.py's
    # module docstring for the confirmed mechanism) - resolve to absolute so callers who pass a
    # project get the directory they actually asked for. `project=None` is left alone: it's an
    # intentional "don't care, use Ultralytics' own default" for one-off ad hoc evaluation.
    if project is not None:
        project = str(Path(project).resolve())
    model = YOLO(str(weights))
    results = model.val(data=str(data_yaml), split=split, project=project, name=name)
    return results, model.names


def split_counts(data_yaml: Path, split: str, results: Any) -> dict:
    """Image/instance counts for a split.

    Total distinct image count ("374" in the README table) is not reachable from `model.val()`'s
    return value - that object (`ultralytics.utils.metrics.DetMetrics`) carries `nt_per_class`
    (instances per class - real detections, safe to sum) and `nt_per_image` (*images containing
    each class* - summing that across classes double-counts any image with >1 damage type), but
    the true total image count only lives on the validator's own `self.seen`, which
    `Model.val()` never exposes publicly (confirmed by reading engine/model.py's `val()`: it
    returns `validator.metrics`, not `validator` itself). So image count is derived independently,
    straight from the dataset the data.yaml points at - works identically for real CarDD splits
    and the tiny synthetic CI fixture.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    split_key = {"train": "train", "val": "val", "test": "test"}[split]
    split_dir = Path(cfg["path"]) / cfg[split_key]
    n_images = sum(1 for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return {
        "images": n_images,
        "instances": int(results.nt_per_class.sum()),
    }
