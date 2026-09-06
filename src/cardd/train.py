"""Canonical training entrypoint - the one code path all four former notebooks collapse into.

Design note: after `model.train()`, this runs two explicit `.val()` calls (split="val" and
split="test") rather than trying to extract training-time val metrics from `model.train()`'s own
return value. That return value's exact `.box`-accessor shape for val metrics wasn't independently
verified against the installed Ultralytics version (unlike `run_evaluation`'s shape, which was:
see evaluate.py). Running val and test symmetrically through the same `run_evaluation()` call
sidesteps that uncertainty entirely and gives both splits the identical, well-defined metrics
shape - at the cost of one redundant validation pass on `best.pt`, which is cheap next to training.

Second design note: `project` is resolved to an absolute path before being passed to
`model.train()`/`model.val()`. Ultralytics' own `get_save_dir()` (ultralytics/cfg/__init__.py)
only uses a `project` argument directly when `Path(project).is_absolute()` - a *relative* project
string is silently reinterpreted as a subdirectory *under a global, machine-level default runs
directory* (`RUNS_DIR/task/project/name`, where `RUNS_DIR` comes from Ultralytics' own persistent
settings, unrelated to this repo). Without resolving to absolute first, `best_weights` below would
point at a different location than whatever Ultralytics actually wrote to, and
`YOLO(best_weights)` would fail with FileNotFoundError - confirmed by hitting exactly that failure
with a relative `--project runs` before this fix was added.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
from ultralytics import YOLO

from cardd.config import ExperimentConfig, build_train_kwargs, flatten_config_for_mlflow
from cardd.evaluate import run_evaluation, split_counts
from cardd.metrics_schema import build_metrics_dict, validate_metrics_schema, write_metrics_json
from cardd.tracking import init_tracking


def run_training(
    cfg: ExperimentConfig,
    project: str = "runs",
    metrics_out: Path | None = None,
) -> dict:
    project = str(Path(project).resolve())
    init_tracking(experiment_name=cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.name, tags=cfg.mlflow_tags):
        mlflow.log_params(flatten_config_for_mlflow(cfg))
        if cfg.source_path:
            mlflow.log_artifact(str(cfg.source_path))

        checkpoint = f"{cfg.model.variant}.{'pt' if cfg.model.pretrained else 'yaml'}"
        model = YOLO(checkpoint)

        start = time.monotonic()
        model.train(
            data=str(cfg.data_yaml),
            project=project,
            name=cfg.name,
            exist_ok=True,
            **build_train_kwargs(cfg),
        )
        train_seconds = time.monotonic() - start

        best_weights = Path(project) / cfg.name / "weights" / "best.pt"

        val_results, class_names = run_evaluation(
            best_weights, cfg.data_yaml, split="val", project=project, name=f"{cfg.name}_val"
        )
        test_results, _ = run_evaluation(
            best_weights, cfg.data_yaml, split="test", project=project, name=f"{cfg.name}_test"
        )
        val_counts = split_counts(cfg.data_yaml, "val", val_results)
        test_counts = split_counts(cfg.data_yaml, "test", test_results)

        active_run = mlflow.active_run()
        metrics = build_metrics_dict(
            run_name=cfg.name,
            config_path=cfg.source_path or "",
            weights_path=best_weights,
            class_names=class_names,
            val_results=val_results,
            test_results=test_results,
            val_counts=val_counts,
            test_counts=test_counts,
            train_seconds=train_seconds,
            mlflow_run_id=active_run.info.run_id if active_run else None,
        )
        validate_metrics_schema(metrics)

        mlflow.log_metrics({
            f"{split_name}_{k}": v
            for split_name, split_data in metrics["splits"].items()
            for k, v in split_data["aggregate"].items()
        })

        out_path = Path(metrics_out) if metrics_out else Path(project) / cfg.name / "metrics.json"
        write_metrics_json(metrics, out_path)
        mlflow.log_artifact(str(out_path))

    return metrics
