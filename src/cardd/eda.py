"""EDA over CarDD_COCO: per-class counts, image size distribution, sample grid with GT boxes."""
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

SPLITS = ["train2017", "val2017", "test2017"]


def load_split(data_root: Path, split: str) -> dict:
    with open(Path(data_root) / "annotations" / f"instances_{split}.json") as f:
        return json.load(f)


def class_counts_table(data_by_split: dict) -> tuple[list[str], dict]:
    id2name = {c["id"]: c["name"] for c in data_by_split["train2017"]["categories"]}
    names = [id2name[i] for i in sorted(id2name)]
    table = {}
    for split, data in data_by_split.items():
        cnt = Counter(a["category_id"] for a in data["annotations"])
        table[split] = {id2name[cid]: cnt.get(cid, 0) for cid in sorted(id2name)}
    return names, table


def plot_class_counts(names, table, out_path, splits=SPLITS):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = range(len(names))
    for i, split in enumerate(splits):
        counts = [table[split][n] for n in names]
        ax.bar([xi + i * width for xi in x], counts, width, label=split)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("annotation count")
    ax.set_title("CarDD class distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_image_sizes(data_by_split, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for split, data in data_by_split.items():
        ws = [img["width"] for img in data["images"]]
        hs = [img["height"] for img in data["images"]]
        ax.scatter(ws, hs, s=8, alpha=0.4, label=split)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image resolution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(data_root: Path, data: dict, split: str, out_path, n=6, seed=0):
    random.seed(seed)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    candidates = [img for img in data["images"] if img["id"] in anns_by_img]
    sample = random.sample(candidates, min(n, len(candidates)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    img_dir = Path(data_root) / split
    for ax, img in zip(axes, sample):
        im = plt.imread(img_dir / img["file_name"])
        ax.imshow(im)
        for a in anns_by_img[img["id"]]:
            x, y, w, h = a["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x, max(y - 4, 0), id2name[a["category_id"]], color="red", fontsize=8,
                     bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"))
        ax.set_title(img["file_name"], fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eda(data_root: Path, out_dir: Path, splits: list[str] = SPLITS) -> dict:
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_by_split = {split: load_split(data_root, split) for split in splits}

    print("=== image / annotation counts ===")
    for split, data in data_by_split.items():
        n_images, n_anns = len(data["images"]), len(data["annotations"])
        print(f"{split:12s} images={n_images:5d}  annotations={n_anns:5d}")

    names, table = class_counts_table(data_by_split)
    print("\n=== per-class counts ===")
    print(f"{'class':15s}" + "".join(f"{s:>12s}" for s in splits))
    for name in names:
        print(f"{name:15s}" + "".join(f"{table[s][name]:12d}" for s in splits))

    plot_class_counts(names, table, out_dir / "class_counts.png", splits=splits)
    plot_image_sizes(data_by_split, out_dir / "image_sizes.png")
    # Filename intentionally matches the pre-refactor scripts/eda.py output exactly
    # ("sample_grid_train.png", not "sample_grid_train2017.png") - that file is already committed.
    plot_sample_grid(
        data_root, data_by_split[splits[0]], splits[0], out_dir / "sample_grid_train.png"
    )

    with open(out_dir / "class_counts.json", "w") as f:
        json.dump(table, f, indent=2)

    print(f"\nSaved plots + class_counts.json to {out_dir}/")
    return table
