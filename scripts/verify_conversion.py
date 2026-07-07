"""Sanity-check the COCO->YOLO conversion: independently decode YOLO-format label txts
(center x/y/w/h, normalized) back to pixel boxes and redraw them, so a bug in this script
would not just mirror a bug in convert_coco_to_yolo.py's own box math."""
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml

DATA_YAML = Path("configs/cardd_yolo.yaml")
OUT_DIR = Path("outputs/sanity")
SPLIT = "train2017"
N_SAMPLES = 6
SEED = 0


def load_names():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["names"], Path(cfg["path"])


def read_yolo_label(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        cls, cx, cy, w, h = line.split()[:5]
        boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def main():
    names, root = load_names()
    img_dir = root / "images" / SPLIT
    lbl_dir = root / "labels" / SPLIT

    random.seed(SEED)
    img_paths = sorted(img_dir.glob("*.jpg"))
    sample = random.sample(img_paths, min(N_SAMPLES, len(img_paths)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, img_path in zip(axes, sample):
        im = plt.imread(img_path)
        img_h, img_w = im.shape[0], im.shape[1]
        ax.imshow(im)
        for cls, cx, cy, w, h in read_yolo_label(lbl_dir / (img_path.stem + ".txt")):
            bw, bh = w * img_w, h * img_h
            x0, y0 = cx * img_w - bw / 2, cy * img_h - bh / 2
            ax.add_patch(patches.Rectangle((x0, y0), bw, bh, linewidth=1.5, edgecolor="lime", facecolor="none"))
            ax.text(x0, max(y0 - 4, 0), names[cls], color="lime", fontsize=8,
                     bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"))
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"yolo_boxes_{SPLIT}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
