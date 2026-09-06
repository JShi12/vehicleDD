"""MLflow experiment-tracking helpers.

Non-obvious behavior this module relies on: Ultralytics' own MLflow callback
(ultralytics/utils/callbacks/mlflow.py) activates automatically the moment `mlflow` is importable.
If *our* code opens the run before calling `model.train()`, Ultralytics attaches to that same run
(logging per-epoch loss/lr and the full weights/ artifact tree for free) and does NOT call
mlflow.end_run() on it - it only closes runs it started itself. So `init_tracking` +
`mlflow.start_run()` must happen before `model.train()`, and our own `with mlflow.start_run():`
block is what actually closes the run afterward.

Second non-obvious behavior: mlflow >=3.x puts the plain local-filesystem tracking backend
(the "mlruns/" directory this project deliberately uses for a zero-infra local tracking store)
into "maintenance mode" and refuses to use it unless `MLFLOW_ALLOW_FILE_STORE=true` is set -
confirmed by hitting `MlflowException: The filesystem tracking backend ... is in maintenance
mode` with mlflow 3.16.0 (not reproduced with the 3.1.4 this was first tested against). mlflow
still supports the file store under that flag - this isn't a workaround for something broken,
it's the officially sanctioned opt-in, kept as the default here rather than migrating to a
database backend (e.g. sqlite) to preserve the "no infra to stand up" design goal.
"""
from __future__ import annotations

import os

import mlflow


def init_tracking(tracking_uri: str = "mlruns", experiment_name: str = "cardd-yolo11") -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
