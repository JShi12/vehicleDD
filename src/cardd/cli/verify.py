"""Sanity-check the COCO->YOLO conversion: independently decode YOLO-format label txts back to
pixel boxes and redraw them, so a bug in this script would not just mirror a bug in convert.py's
own box math."""
import argparse
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml

from cardd.data.yolo_format import read_yolo_label, yolo_box_to_pixels


def verify_conversion(
    data_yaml: Path, out_dir: Path, split: str = "train2017", n_samples: int = 6, seed: int = 0
):
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names, root = cfg["names"], Path(cfg["path"])
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    random.seed(seed)
    img_paths = sorted(img_dir.glob("*.jpg"))
    sample = random.sample(img_paths, min(n_samples, len(img_paths)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, img_path in zip(axes, sample):
        im = plt.imread(img_path)
        img_h, img_w = im.shape[0], im.shape[1]
        ax.imshow(im)
        for cls, cx, cy, w, h in read_yolo_label(lbl_dir / (img_path.stem + ".txt")):
            x0, y0, bw, bh = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
            ax.add_patch(patches.Rectangle(
                (x0, y0), bw, bh, linewidth=1.5, edgecolor="lime", facecolor="none"
            ))
            ax.text(x0, max(y0 - 4, 0), names[cls], color="lime", fontsize=8,
                     bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"))
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"yolo_boxes_{split}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-yaml", type=Path, default=Path("configs/cardd_yolo.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/sanity"))
    p.add_argument("--split", default="train2017")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    verify_conversion(args.data_yaml, args.out_dir, args.split, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
