import pytest

from cardd.metrics_schema import build_metrics_dict, build_split_metrics, validate_metrics_schema


class FakeBox:
    """Stands in for ultralytics.utils.metrics.DetMetrics.box - only the attributes
    build_split_metrics actually reads, so this test is independent of any real trained model."""

    def __init__(self, mp, mr, map50, map_, ap_class_index, per_class, f1):
        self.mp = mp
        self.mr = mr
        self.map50 = map50
        self.map = map_
        self.ap_class_index = ap_class_index
        self._per_class = per_class
        self.f1 = f1

    def class_result(self, i):
        return self._per_class[i]


class FakeResults:
    def __init__(self, box, nt_per_class):
        self.box = box
        self.nt_per_class = nt_per_class


def make_fake_results():
    box = FakeBox(
        mp=0.783, mr=0.685, map50=0.728, map_=0.568,
        ap_class_index=[0, 1],
        per_class=[(0.7, 0.5, 0.6, 0.3), (0.9, 0.8, 0.9, 0.7)],
        f1=[0.58, 0.85],
    )
    return FakeResults(box, nt_per_class=[10, 20])


CLASS_NAMES = {0: "dent", 1: "scratch"}


def test_build_split_metrics_shape():
    results = make_fake_results()
    split = build_split_metrics(results.box, CLASS_NAMES)

    assert split["aggregate"] == {
        "precision": 0.783, "recall": 0.685, "map50": 0.728, "map50_95": 0.568,
    }
    assert len(split["per_class"]) == 2
    assert split["per_class"][0] == {
        "class_id": 0, "class_name": "dent", "precision": 0.7, "recall": 0.5,
        "f1": 0.58, "ap50": 0.6, "ap50_95": 0.3,
    }


def test_build_metrics_dict_requires_at_least_one_split():
    with pytest.raises(ValueError):
        build_metrics_dict("run", "cfg.yaml", "best.pt", CLASS_NAMES)


def test_build_metrics_dict_and_validate_round_trip():
    val_results = make_fake_results()
    test_results = make_fake_results()

    metrics = build_metrics_dict(
        run_name="cardd_yolo11n",
        config_path="configs/experiments/01_baseline.yaml",
        weights_path="runs/cardd_yolo11n/weights/best.pt",
        class_names=CLASS_NAMES,
        val_results=val_results,
        test_results=test_results,
        val_counts={"images": 10, "instances": 30},
        test_counts={"images": 5, "instances": 15},
    )

    validate_metrics_schema(metrics)
    assert metrics["class_names"] == ["dent", "scratch"]
    assert metrics["splits"]["val"]["images"] == 10
    assert metrics["splits"]["test"]["aggregate"]["map50_95"] == 0.568


def test_validate_metrics_schema_rejects_missing_keys():
    with pytest.raises(AssertionError):
        validate_metrics_schema({"schema_version": 1, "splits": {"test": {"aggregate": {}}}})
