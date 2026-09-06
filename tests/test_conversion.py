import pytest

from cardd.data.coco import extract_class_names
from cardd.data.convert import convert_coco_to_yolo
from cardd.data.yolo_format import read_yolo_label
from tests.fixtures.synthetic_dataset import IMG_SIZE, KNOWN_BBOX, build_synthetic_coco_dataset

CARDD_CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]


def test_known_bbox_round_trips_through_conversion(tmp_path):
    coco_root = build_synthetic_coco_dataset(tmp_path / "coco", counts={"train2017": 1})
    out_root = tmp_path / "cardd_yolo"
    data_yaml = tmp_path / "cardd_yolo.yaml"

    convert_coco_to_yolo(coco_root, out_root, data_yaml, splits=["train2017"], expected_counts={})

    label_files = sorted((out_root / "labels" / "train2017").glob("*.txt"))
    assert len(label_files) == 1

    boxes = read_yolo_label(label_files[0])
    assert len(boxes) == 1

    _cls, cx, cy, w, h = boxes[0]
    x, y, bw, bh = KNOWN_BBOX
    expected_cx = (x + bw / 2) / IMG_SIZE
    expected_cy = (y + bh / 2) / IMG_SIZE
    expected_w = bw / IMG_SIZE
    expected_h = bh / IMG_SIZE
    assert cx == pytest.approx(expected_cx, abs=1e-4)
    assert cy == pytest.approx(expected_cy, abs=1e-4)
    assert w == pytest.approx(expected_w, abs=1e-4)
    assert h == pytest.approx(expected_h, abs=1e-4)


def test_convert_writes_data_yaml_with_class_names(tmp_path):
    coco_root = build_synthetic_coco_dataset(tmp_path / "coco", counts={"train2017": 2})
    out_root = tmp_path / "cardd_yolo"
    data_yaml = tmp_path / "cardd_yolo.yaml"

    result = convert_coco_to_yolo(
        coco_root, out_root, data_yaml, splits=["train2017"], expected_counts={}
    )

    assert result.class_names == CARDD_CLASS_NAMES
    assert data_yaml.exists()


def test_extract_class_names_raises_on_noncontiguous_ids():
    categories = [
        {"id": 1, "name": "dent"}, {"id": 2, "name": "scratch"}, {"id": 4, "name": "crack"},
    ]
    with pytest.raises(AssertionError):
        extract_class_names(categories)


def test_extract_class_names_sorts_by_id():
    categories = [{"id": 2, "name": "b"}, {"id": 1, "name": "a"}, {"id": 3, "name": "c"}]
    assert extract_class_names(categories) == ["a", "b", "c"]
