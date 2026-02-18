"""VOC 与 YOLO 标注转换工具。"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ez_traing.data_prep.models import AnnotationBox, DatasetSample

_VOC_RAW_CACHE: Dict[Tuple[str, int], List[Tuple[str, float, float, float, float]]] = {}


def load_existing_classes(dataset_root: Path) -> List[str]:
    """读取已有 classes.txt。"""
    for path in [dataset_root / "classes.txt", dataset_root / "labels" / "classes.txt"]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            if classes:
                return classes
    return []


def save_classes(classes_path: Path, class_names: List[str]) -> None:
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")


def find_voc_for_image(image_path: Path, dataset_root: Path) -> Optional[Path]:
    """为图片定位 VOC 标注文件。"""
    candidate = image_path.with_suffix(".xml")
    if candidate.exists():
        return candidate

    alt = dataset_root / "Annotations" / f"{image_path.stem}.xml"
    if alt.exists():
        return alt

    return None


def parse_voc_boxes(xml_path: Path, image_width: int, image_height: int) -> List[AnnotationBox]:
    """解析 VOC XML 框。"""
    xml_path = xml_path.resolve()
    try:
        mtime_ns = xml_path.stat().st_mtime_ns
    except OSError as e:
        raise ValueError(f"读取 XML 失败: {xml_path} ({e})") from e

    cache_key = (str(xml_path), mtime_ns)
    cached_raw = _VOC_RAW_CACHE.get(cache_key)

    if cached_raw is None:
        # 删除该路径旧版本缓存，避免缓存无限增长。
        stale_keys = [k for k in _VOC_RAW_CACHE.keys() if k[0] == str(xml_path) and k != cache_key]
        for key in stale_keys:
            _VOC_RAW_CACHE.pop(key, None)

        raw_boxes: List[Tuple[str, float, float, float, float]] = []
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as e:
            raise ValueError(f"解析 XML 失败: {xml_path} ({e})") from e

        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if not label or bnd is None:
                continue

            try:
                x_min = float((bnd.findtext("xmin") or "0").strip())
                y_min = float((bnd.findtext("ymin") or "0").strip())
                x_max = float((bnd.findtext("xmax") or "0").strip())
                y_max = float((bnd.findtext("ymax") or "0").strip())
            except ValueError:
                continue
            raw_boxes.append((label, x_min, y_min, x_max, y_max))

        _VOC_RAW_CACHE[cache_key] = raw_boxes
        cached_raw = raw_boxes

    boxes: List[AnnotationBox] = []
    for label, x_min, y_min, x_max, y_max in cached_raw:
        x_min = max(0.0, min(x_min, float(image_width)))
        x_max = max(0.0, min(x_max, float(image_width)))
        y_min = max(0.0, min(y_min, float(image_height)))
        y_max = max(0.0, min(y_max, float(image_height)))

        if x_max <= x_min or y_max <= y_min:
            continue

        boxes.append(
            AnnotationBox(
                label=label,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    return boxes


def build_class_names(samples: Iterable[DatasetSample], existing_classes: List[str]) -> List[str]:
    """构建类别列表，优先沿用已有 classes。"""
    names = list(existing_classes)
    seen = set(names)
    for sample in samples:
        for box in sample.boxes:
            if box.label not in seen:
                names.append(box.label)
                seen.add(box.label)
    return names


def write_yolo_label(
    label_path: Path,
    boxes: List[AnnotationBox],
    class_to_id: Dict[str, int],
    image_width: int,
    image_height: int,
) -> None:
    """将框写入 YOLO txt。"""
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for box in boxes:
        class_id = class_to_id.get(box.label)
        if class_id is None:
            continue

        x_center = ((box.x_min + box.x_max) / 2.0) / image_width
        y_center = ((box.y_min + box.y_max) / 2.0) / image_height
        width = (box.x_max - box.x_min) / image_width
        height = (box.y_max - box.y_min) / image_height

        if width <= 0.0 or height <= 0.0:
            continue

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    with open(label_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
