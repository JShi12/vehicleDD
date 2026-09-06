import argparse
from pathlib import Path

from cardd.data.convert import convert_coco_to_yolo


def main():
    p = argparse.ArgumentParser(description="Convert CarDD_COCO annotations to YOLO format.")
    p.add_argument("--coco-root", type=Path, default=Path("data/CarDD_release/CarDD_COCO"))
    p.add_argument("--out-root", type=Path, default=Path("data/cardd_yolo"))
    p.add_argument("--data-yaml", type=Path, default=Path("configs/cardd_yolo.yaml"))
    args = p.parse_args()

    result = convert_coco_to_yolo(args.coco_root, args.out_root, args.data_yaml)
    print("Class names (from dataset JSON):", result.class_names)
    print(f"Wrote {result.data_yaml_path}")
    print(f"{'split':12s}{'images':>10s}{'label_files':>14s}")
    for split, n_images in result.image_counts.items():
        print(f"{split:12s}{n_images:10d}{result.label_counts[split]:14d}")


if __name__ == "__main__":
    main()
