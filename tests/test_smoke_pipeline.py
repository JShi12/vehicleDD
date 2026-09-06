"""Full pipeline smoke test: synthetic dataset -> convert -> 1-epoch CPU train (random-init,
no network) -> eval -> metrics.json. This is the CI centerpiece proving the refactored pipeline
actually works end-to-end, without ever touching real (license-restricted) CarDD data or a
pretrained-checkpoint download."""
from pathlib import Path

from cardd.config import ExperimentConfig, ModelConfig, TrainConfig
from cardd.data.convert import convert_coco_to_yolo
from cardd.metrics_schema import validate_metrics_schema
from cardd.train import run_training


def test_full_pipeline_smoke(tmp_path, synthetic_coco_root, monkeypatch):
    # keep mlflow's default relative "mlruns" store contained to tmp_path
    monkeypatch.chdir(tmp_path)

    out_root = tmp_path / "cardd_yolo"
    data_yaml = tmp_path / "configs" / "cardd_yolo.yaml"
    convert_coco_to_yolo(synthetic_coco_root, out_root, data_yaml, expected_counts={})

    cfg = ExperimentConfig(
        name="ci_smoke",
        data_yaml=data_yaml,
        model=ModelConfig(variant="yolo11n", pretrained=False),
        train=TrainConfig(epochs=1, imgsz=32, batch=2, device="cpu"),
        mlflow_experiment="cardd-ci-smoke",
    )

    metrics_out = tmp_path / "metrics.json"
    metrics = run_training(cfg, project=str(tmp_path / "runs"), metrics_out=metrics_out)

    validate_metrics_schema(metrics)
    assert metrics_out.exists()
    expected_names = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
    assert metrics["class_names"] == expected_names
    assert set(metrics["splits"]) == {"val", "test"}
    assert (Path(tmp_path / "runs" / "ci_smoke" / "weights" / "best.pt")).exists()
