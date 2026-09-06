"""COCO annotation loading and class-name extraction.

Split into IO (`load_categories`) and pure logic (`extract_class_names`) so the
contiguous-category-id assertion - required for `cls91to80=False` in convert.py to map category
ids to class indices correctly - is unit-testable without touching disk.
"""
import json
from pathlib import Path


def load_categories(coco_root: Path, split: str = "train2017") -> list[dict]:
    ann_path = Path(coco_root) / "annotations" / f"instances_{split}.json"
    with open(ann_path) as f:
        return json.load(f)["categories"]


def extract_class_names(categories: list[dict]) -> list[str]:
    ids = sorted(c["id"] for c in categories)
    assert ids == list(range(1, len(ids) + 1)), (
        f"Expected contiguous category ids starting at 1 (required for cls91to80=False to map "
        f"correctly), got {ids}."
    )
    return [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
