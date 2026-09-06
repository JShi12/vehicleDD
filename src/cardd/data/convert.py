"""Convert CarDD_COCO-format annotations to YOLO format and assemble the images/labels layout
Ultralytics expects.

Note: convert_coco()'s default cls91to80=True remaps category ids through the standard
91->80 COCO class table, which is meaningless for CarDD's own custom classes. It must
be False here so category ids map straight to class indices (id - 1).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics.data.converter import convert_coco

from cardd.data.coco import extract_class_names, load_categories

SPLITS = ["train2017", "val2017", "test2017"]
EXPECTED_COUNTS = {"train2017": 2816, "val2017": 810, "test2017": 374}


@dataclass
class ConversionResult:
    out_root: Path
    data_yaml_path: Path
    class_names: list[str]
    image_counts: dict[str, int]
    label_counts: dict[str, int]


def convert_coco_to_yolo(
    coco_root: Path,
    out_root: Path,
    data_yaml_path: Path,
    splits: list[str] = SPLITS,
    expected_counts: dict[str, int] | None = None,
) -> ConversionResult:
    coco_root, out_root, data_yaml_path = Path(coco_root), Path(out_root), Path(data_yaml_path)

    class_names = extract_class_names(load_categories(coco_root))

    if out_root.exists():
        shutil.rmtree(out_root)  # convert_coco increments the dir name instead of overwriting
    convert_coco(
        labels_dir=str(coco_root / "annotations"),
        save_dir=str(out_root),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

    for split in splits:
        src_dir = coco_root / split
        dst_dir = out_root / "images" / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.glob("*.jpg"):
            dst = dst_dir / img_path.name
            if not dst.exists():
                dst.symlink_to(img_path.resolve())

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(
            {
                "path": str(out_root.resolve()),
                "train": "images/train2017",
                "val": "images/val2017",
                "test": "images/test2017",
                "names": {i: n for i, n in enumerate(class_names)},
            },
            f,
            sort_keys=False,
        )

    image_counts, label_counts = {}, {}
    for split in splits:
        image_counts[split] = len(list((out_root / "images" / split).glob("*.jpg")))
        label_counts[split] = len(list((out_root / "labels" / split).glob("*.txt")))
        expected = (expected_counts or EXPECTED_COUNTS).get(split)
        if expected is not None and image_counts[split] != expected:
            print(
                f"  WARNING: expected {expected} images for split '{split}', found "
                f"{image_counts[split]} - check the download/extraction completed correctly."
            )

    return ConversionResult(out_root, data_yaml_path, class_names, image_counts, label_counts)
