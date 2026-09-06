import argparse
from pathlib import Path

from cardd.config import load_experiment_config
from cardd.train import run_training


def main():
    p = argparse.ArgumentParser(description="Train a CarDD YOLO11 experiment from a config file.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--project", default="runs")
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="key.subkey=value",
        help="Override a config value, e.g. --set train.device=cpu --set train.epochs=1",
    )
    args = p.parse_args()

    cfg = load_experiment_config(args.config, overrides=args.overrides)
    metrics = run_training(cfg, project=args.project, metrics_out=args.metrics_out)
    print(f"\ntest mAP50-95: {metrics['splits']['test']['aggregate']['map50_95']:.4f}")


if __name__ == "__main__":
    main()
