"""Shared annotation parsing utilities for YOLO and VOC formats."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

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
    labels: List[str] = []
    try:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name = obj.find("name")
            if name is not None and name.text:
                labels.append(name.text)
    except Exception:
        logger.debug("Failed to parse VOC annotation: %s", xml_path)
    return labels


def read_yolo_boxes(
    txt_path: Path, img_w: int, img_h: int, class_names: List[str],
) -> List[Dict[str, object]]:
    """Parse a YOLO annotation file and return bounding boxes.

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
                except ValueError:
                    continue
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                if class_names and 0 <= class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = f"class_{class_id}"
                boxes.append({
                    "label": label,
                    "xmin": int(cx - w / 2),
                    "ymin": int(cy - h / 2),
                    "xmax": int(cx + w / 2),
                    "ymax": int(cy + h / 2),
                })
    except Exception:
        logger.debug("Failed to parse YOLO boxes: %s", txt_path)
    return boxes


def read_voc_boxes(xml_path: Path) -> List[Dict[str, object]]:
    """Parse a VOC XML annotation file and return bounding boxes.

    Returns list of ``{"label": str, "xmin": int, "ymin": int,
    "xmax": int, "ymax": int}``.
    """
    boxes: List[Dict[str, object]] = []
    try:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if not name or bnd is None:
                continue
            boxes.append({
                "label": name,
                "xmin": int(float((bnd.findtext("xmin") or "0").strip())),
                "ymin": int(float((bnd.findtext("ymin") or "0").strip())),
                "xmax": int(float((bnd.findtext("xmax") or "0").strip())),
                "ymax": int(float((bnd.findtext("ymax") or "0").strip())),
            })
    except Exception:
        logger.debug("Failed to parse VOC boxes: %s", xml_path)
    return boxes


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
