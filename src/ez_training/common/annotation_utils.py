"""Shared annotation parsing utilities for YOLO and VOC formats."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ez_training.common.voc_io import parse_voc_objects

logger = logging.getLogger(__name__)


def parse_yolo_labels(txt_path: Path, class_names: List[str]) -> List[str]:
    """Parse a YOLO annotation file and return label names.

    Does not require image dimensions since only the class index is used.
    """
    labels: List[str] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                if class_names and 0 <= class_id < len(class_names):
                    labels.append(class_names[class_id])
                else:
                    labels.append(f"class_{class_id}")
    except Exception:
        logger.debug("Failed to parse YOLO annotation: %s", txt_path)
    return labels


def parse_voc_labels(xml_path: Path) -> List[str]:
    """Parse a VOC XML annotation file and return label names."""
    return [o.label for o in parse_voc_objects(xml_path)]


def read_yolo_boxes(
    txt_path: Path, img_w: int, img_h: int, class_names: List[str],
) -> List[Dict[str, object]]:
    """Parse a YOLO annotation file and return bounding boxes.

    Coordinates are clamped to ``[0, img_w]`` / ``[0, img_h]`` and
    degenerate boxes (width or height <= 0) are skipped.

    Returns list of ``{"label": str, "xmin": int, "ymin": int,
    "xmax": int, "ymax": int}``.
    """
    boxes: List[Dict[str, object]] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                except (ValueError, IndexError):
                    continue
                if class_names and 0 <= class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = f"class_{class_id}"
                xmin = max(0, round(cx - w / 2))
                ymin = max(0, round(cy - h / 2))
                xmax = min(img_w, round(cx + w / 2))
                ymax = min(img_h, round(cy + h / 2))
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append({
                    "label": label,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                })
    except Exception:
        logger.debug("Failed to parse YOLO boxes: %s", txt_path)
    return boxes


def read_voc_boxes(xml_path: Path) -> List[Dict[str, object]]:
    """Parse a VOC XML annotation file and return bounding boxes.

    Returns list of ``{"label": str, "xmin": int, "ymin": int,
    "xmax": int, "ymax": int}``.
    """
    return [
        {"label": o.label, "xmin": o.xmin, "ymin": o.ymin,
         "xmax": o.xmax, "ymax": o.ymax}
        for o in parse_voc_objects(xml_path)
    ]


def read_annotation_boxes(
    image_path: str,
    img_w: int,
    img_h: int,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Read annotation boxes for an image (YOLO priority, fallback to VOC).

    Priority: if a ``.txt`` file exists and contains valid annotations,
    use YOLO format; otherwise try VOC ``.xml``.
    """
    path = Path(image_path)
    names = class_names or []

    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        boxes = read_yolo_boxes(txt_path, img_w, img_h, names)
        if boxes:
            return boxes

    xml_path = path.with_suffix(".xml")
    if xml_path.exists():
        return read_voc_boxes(xml_path)

    return []
