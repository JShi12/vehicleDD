"""Procedurally build a tiny COCO-format dataset mirroring CarDD's real schema, for tests only.

CarDD's license forbids redistributing its data, so nothing real can ever be committed as a
fixture - this generates a synthetic dataset fresh at test time instead. Schema (top-level
licenses/info/categories/images/annotations; categories:[{id,name}] ids 1-6 contiguous;
images:[{id,width,height,file_name,license}]; annotations:[{id,image_id,category_id,segmentation,
area,bbox,iscrowd,attributes}]) was verified against the real CarDD_COCO json files on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

CATEGORIES = [
    {"id": 1, "name": "dent"},
    {"id": 2, "name": "scratch"},
    {"id": 3, "name": "crack"},
    {"id": 4, "name": "glass shatter"},
    {"id": 5, "name": "lamp broken"},
    {"id": 6, "name": "tire flat"},
]

IMG_SIZE = 64
# Same bbox (COCO x, y, w, h in pixels) on every image, so any test can assert on one constant
# regardless of which generated image it happens to inspect.
KNOWN_BBOX = [8.0, 8.0, 20.0, 20.0]

DEFAULT_COUNTS = {"train2017": 4, "val2017": 2, "test2017": 2}


def build_synthetic_coco_dataset(root: Path, counts: dict[str, int] | None = None) -> Path:
    root = Path(root)
    counts = DEFAULT_COUNTS if counts is None else counts
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    image_id = 1
    ann_id = 1
    for split, n in counts.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for i in range(n):
            file_name = f"{split}_{i:03d}.jpg"
            color = ((image_id * 37) % 255, (image_id * 61) % 255, (image_id * 89) % 255)
            Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=color).save(split_dir / file_name)
            images.append({
                "id": image_id, "width": IMG_SIZE, "height": IMG_SIZE,
                "file_name": file_name, "license": 1,
            })

            category_id = CATEGORIES[image_id % len(CATEGORIES)]["id"]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],
                "area": KNOWN_BBOX[2] * KNOWN_BBOX[3],
                "bbox": list(KNOWN_BBOX),
                "iscrowd": 0,
                "attributes": {},
            })
            ann_id += 1
            image_id += 1

        with open(ann_dir / f"instances_{split}.json", "w") as f:
            json.dump({
                "licenses": [{"id": 1, "name": "synthetic-test-fixture", "url": ""}],
                "info": {"description": "synthetic fixture, not real CarDD data", "version": "1.0"},
                "categories": CATEGORIES,
                "images": images,
                "annotations": annotations,
            }, f)

    return root
