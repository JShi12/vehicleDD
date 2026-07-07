"""Convert CarDD_COCO annotations to YOLO format and assemble the images/labels layout Ultralytics expects.

Note: convert_coco()'s default cls91to80=True remaps category ids through the standard
91->80 COCO class table, which is meaningless for CarDD's own custom classes. It must
be False here so category ids map straight to class indices (id - 1).

Class names are read from the dataset's own JSON rather than hardcoded, so a mismatch
(e.g. after a re-download or version bump) fails loudly instead of silently mislabeling classes.
"""
import json
import shutil
from pathlib import Path

import yaml
from ultralytics.data.converter import convert_coco

COCO_ROOT = Path("data/CarDD_release/CarDD_COCO")
OUT_ROOT = Path("data/cardd_yolo")
SPLITS = ["train2017", "val2017", "test2017"]
EXPECTED_COUNTS = {"train2017": 2816, "val2017": 810, "test2017": 374}


def get_class_names():
    with open(COCO_ROOT / "annotations" / "instances_train2017.json") as f:
        categories = json.load(f)["categories"]
    ids = sorted(c["id"] for c in categories)
    assert ids == list(range(1, len(ids) + 1)), (
        f"Expected contiguous category ids starting at 1 (required for cls91to80=False to map "
        f"correctly), got {ids}."
    )
    return [c["name"] for c in sorted(categories, key=lambda c: c["id"])]


def convert_labels():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)  # convert_coco increments the dir name instead of overwriting
    convert_coco(
        labels_dir=str(COCO_ROOT / "annotations"),
        save_dir=str(OUT_ROOT),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )


def link_images():
    for split in SPLITS:
        src_dir = COCO_ROOT / split
        dst_dir = OUT_ROOT / "images" / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.glob("*.jpg"):
            dst = dst_dir / img_path.name
            if not dst.exists():
                dst.symlink_to(img_path.resolve())


def write_data_yaml(class_names):
    data = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "test": "images/test2017",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    out_path = Path("configs/cardd_yolo.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Wrote {out_path}")


def verify_counts():
    print(f"{'split':12s}{'images':>10s}{'label_files':>14s}")
    for split in SPLITS:
        n_images = len(list((OUT_ROOT / "images" / split).glob("*.jpg")))
        n_labels = len(list((OUT_ROOT / "labels" / split).glob("*.txt")))
        print(f"{split:12s}{n_images:10d}{n_labels:14d}")
        if n_images != EXPECTED_COUNTS[split]:
            print(f"  WARNING: expected {EXPECTED_COUNTS[split]} images for split '{split}', found "
                  f"{n_images} - check the download/extraction completed correctly.")


def main():
    class_names = get_class_names()
    print("Class names (from dataset JSON):", class_names)
    convert_labels()
    link_images()
    write_data_yaml(class_names)
    verify_counts()


if __name__ == "__main__":
    main()
