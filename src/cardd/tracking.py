"""MLflow experiment-tracking helpers.

Non-obvious behavior this module relies on: Ultralytics' own MLflow callback
(ultralytics/utils/callbacks/mlflow.py) activates automatically the moment `mlflow` is importable.
If *our* code opens the run before calling `model.train()`, Ultralytics attaches to that same run
(logging per-epoch loss/lr and the full weights/ artifact tree for free) and does NOT call
mlflow.end_run() on it - it only closes runs it started itself. So `init_tracking` +
`mlflow.start_run()` must happen before `model.train()`, and our own `with mlflow.start_run():`
block is what actually closes the run afterward.
"""
from __future__ import annotations

import mlflow


def init_tracking(tracking_uri: str = "mlruns", experiment_name: str = "cardd-yolo11") -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
