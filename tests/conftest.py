from pathlib import Path

import pytest

from tests.fixtures.synthetic_dataset import build_synthetic_coco_dataset


@pytest.fixture
def synthetic_coco_root(tmp_path: Path) -> Path:
    """A full train/val/test synthetic COCO dataset, for pipeline-level tests."""
    return build_synthetic_coco_dataset(tmp_path / "CarDD_COCO")
