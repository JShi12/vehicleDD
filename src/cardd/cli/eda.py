import argparse
from pathlib import Path

from cardd.eda import run_eda


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/CarDD_release/CarDD_COCO"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eda"))
    args = p.parse_args()
    run_eda(args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
