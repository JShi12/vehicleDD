import argparse
from pathlib import Path

from cardd.evaluate import run_evaluation, split_counts
from cardd.metrics_schema import build_metrics_dict, validate_metrics_schema, write_metrics_json


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint against a data split.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data-yaml", type=Path, default=Path("configs/cardd_yolo.yaml"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--run-name", default=None)
    p.add_argument("--metrics-out", type=Path, default=Path("metrics.json"))
    p.add_argument("--project", default="runs", help="Where Ultralytics writes its own plots")
    args = p.parse_args()

    run_name = args.run_name or args.weights.stem
    results, class_names = run_evaluation(
        args.weights, args.data_yaml, split=args.split,
        project=args.project, name=f"{run_name}_{args.split}",
    )
    counts = split_counts(args.data_yaml, args.split, results)

    kwargs = {f"{args.split}_results": results, f"{args.split}_counts": counts}
    metrics = build_metrics_dict(
        run_name=run_name,
        config_path=args.data_yaml,
        weights_path=args.weights,
        class_names=class_names,
        **kwargs,
    )
    validate_metrics_schema(metrics)
    write_metrics_json(metrics, args.metrics_out)

    agg = metrics["splits"][args.split]["aggregate"]
    print(f"{args.split}: precision={agg['precision']:.3f} recall={agg['recall']:.3f} "
          f"mAP50={agg['map50']:.3f} mAP50-95={agg['map50_95']:.3f}")
    print(f"Wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
