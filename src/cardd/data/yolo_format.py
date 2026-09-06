"""Shared YOLO label decode logic - used by both the visual `cardd-verify` sanity CLI and the
automated conversion tests, so a decode bug is caught by the tests even though the plotting CLI
itself has no assertions."""
from pathlib import Path


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Decode a YOLO-format label file: each line is `cls cx cy w h`, normalized to [0, 1]."""
    path = Path(path)
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        cls, cx, cy, w, h = line.split()[:5]
        boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def yolo_box_to_pixels(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int):
    """Decode a normalized center-x/y/w/h box to pixel (x0, y0, box_w, box_h)."""
    bw, bh = w * img_w, h * img_h
    x0, y0 = cx * img_w - bw / 2, cy * img_h - bh / 2
    return x0, y0, bw, bh
