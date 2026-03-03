"""自动检测游戏截图中敌人血条并生成 VOC 标注。

基于 MaaEnd AutoFight 的 hasEnemyInScreen() 逻辑，通过 RGB 颜色匹配 +
连通域分析在两个 ROI 区域（小怪中部区 / Boss 顶部区）检测红色血条。

所有 ROI 坐标与过滤阈值以 1280x720 为基准，运行时按实际分辨率自动缩放。
"""

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

LABEL = "血条"

# ---------------------------------------------------------------------------
# 基准分辨率 & 检测区域定义（坐标/阈值均基于 1280x720）
# ---------------------------------------------------------------------------

REF_WIDTH = 1280
REF_HEIGHT = 720


@dataclass(frozen=True)
class HpBarZone:
    name: str
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int
    lower_rgb: Tuple[int, int, int]
    upper_rgb: Tuple[int, int, int]
    min_pixel_count: int


ZONES: List[HpBarZone] = [
    HpBarZone(
        name="小怪血条",
        roi_x=0, roi_y=0, roi_w=1280, roi_h=620,
        lower_rgb=(235, 75, 95),
        upper_rgb=(255, 95, 120),
        min_pixel_count=50,
    ),
    HpBarZone(
        name="Boss血条",
        roi_x=400, roi_y=0, roi_w=500, roi_h=100,
        lower_rgb=(240, 60, 90),
        upper_rgb=(255, 90, 140),
        min_pixel_count=200,
    ),
]

MIN_ASPECT_RATIO = 2.0
REF_MIN_WIDTH = 15
REF_MORPH_KERNEL = 3
REF_BOX_PADDING = 3


# ---------------------------------------------------------------------------
# 缩放工具
# ---------------------------------------------------------------------------

def _scale_roi(
    zone: HpBarZone, sx: float, sy: float, img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """将基准 ROI 按缩放因子映射到实际图片，clamp 到图片边界。"""
    x = max(0, int(zone.roi_x * sx))
    y = max(0, int(zone.roi_y * sy))
    w = int(zone.roi_w * sx)
    h = int(zone.roi_h * sy)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    return x, y, x2, y2


def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> np.ndarray:
    return np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 血条检测
# ---------------------------------------------------------------------------

def detect_hp_bars(image_path: str) -> List[Tuple[int, int, int, int]]:
    """检测图片中的敌人血条。

    Returns:
        [(xmin, ymin, xmax, ymax), ...] 原图坐标系下的外接矩形列表。
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  无法读取图片: {image_path}")
        return []

    img_h, img_w = img.shape[:2]
    sx = img_w / REF_WIDTH
    sy = img_h / REF_HEIGHT
    area_scale = sx * sy

    kernel_size = max(1, int(REF_MORPH_KERNEL * min(sx, sy)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    min_width = int(REF_MIN_WIDTH * sx)
    pad_x = max(1, int(REF_BOX_PADDING * sx))
    pad_y = max(1, int(REF_BOX_PADDING * sy))

    all_bars: List[Tuple[int, int, int, int]] = []

    for zone in ZONES:
        x1, y1, x2, y2 = _scale_roi(zone, sx, sy, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = img[y1:y2, x1:x2]

        lower_bgr = _rgb_to_bgr(zone.lower_rgb)
        upper_bgr = _rgb_to_bgr(zone.upper_rgb)
        mask = cv2.inRange(roi, lower_bgr, upper_bgr)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        scaled_min_count = int(zone.min_pixel_count * area_scale)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < scaled_min_count:
                continue

            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            if bh == 0 or bw / bh < MIN_ASPECT_RATIO:
                continue
            if bw < min_width:
                continue

            xmin = max(0, x1 + bx - pad_x)
            ymin = max(0, y1 + by - pad_y)
            xmax = min(img_w, x1 + bx + bw + pad_x)
            ymax = min(img_h, y1 + by + bh + pad_y)
            all_bars.append((xmin, ymin, xmax, ymax))

    return all_bars


# ---------------------------------------------------------------------------
# 图片尺寸
# ---------------------------------------------------------------------------

def _get_image_size(image_path: str) -> Tuple[int, int, int]:
    """返回 (width, height, depth)。"""
    with Image.open(image_path) as img:
        w, h = img.size
        mode_depth = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
        depth = mode_depth.get(img.mode, 3)
    return w, h, depth


# ---------------------------------------------------------------------------
# VOC XML 读写
# ---------------------------------------------------------------------------

def _create_voc_root(
    folder: str, filename: str, path: str, width: int, height: int, depth: int,
) -> ET.Element:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = path

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size_elem = ET.SubElement(root, "size")
    ET.SubElement(size_elem, "width").text = str(width)
    ET.SubElement(size_elem, "height").text = str(height)
    ET.SubElement(size_elem, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = "0"
    return root


def _append_object(
    root: ET.Element, label: str, xmin: int, ymin: int, xmax: int, ymax: int,
) -> None:
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = label
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)


def _read_existing_objects(
    xml_path: str,
) -> List[Tuple[str, int, int, int, int]]:
    """读取已有 VOC XML 中的所有目标框。"""
    if not os.path.exists(xml_path):
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        print(f"  VOC 文件解析失败，将覆盖: {xml_path}")
        return []

    objects: List[Tuple[str, int, int, int, int]] = []
    for obj_elem in root.findall("object"):
        name = (obj_elem.findtext("name") or "").strip()
        bnd = obj_elem.find("bndbox")
        if not name or bnd is None:
            continue
        try:
            xmin = int(float((bnd.findtext("xmin") or "0").strip()))
            ymin = int(float((bnd.findtext("ymin") or "0").strip()))
            xmax = int(float((bnd.findtext("xmax") or "0").strip()))
            ymax = int(float((bnd.findtext("ymax") or "0").strip()))
        except ValueError:
            continue
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects


def _save_voc(
    image_path: str, bars: List[Tuple[int, int, int, int]],
) -> Tuple[str, int]:
    """保存或合并 VOC 标注，返回 (xml_path, 新增框数)。"""
    img_path = Path(image_path)
    xml_path = str(img_path.with_suffix(".xml"))

    width, height, depth = _get_image_size(image_path)
    folder = img_path.parent.name
    filename = img_path.name

    existing_objects = _read_existing_objects(xml_path)

    root = _create_voc_root(folder, filename, str(img_path), width, height, depth)

    added_keys: set = set()
    for name, xmin, ymin, xmax, ymax in existing_objects:
        key = (name, xmin, ymin, xmax, ymax)
        if key not in added_keys:
            _append_object(root, name, xmin, ymin, xmax, ymax)
            added_keys.add(key)

    new_count = 0
    for xmin, ymin, xmax, ymax in bars:
        key = (LABEL, xmin, ymin, xmax, ymax)
        if key not in added_keys:
            _append_object(root, LABEL, xmin, ymin, xmax, ymax)
            added_keys.add(key)
            new_count += 1

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    return xml_path, new_count


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run(dataset_dir: str) -> None:
    print(f"开始处理数据集: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        print(f"错误: 目录不存在 - {dataset_dir}")
        return

    image_files = sorted(
        f for f in Path(dataset_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"找到 {len(image_files)} 张图片（含子目录）")
    if not image_files:
        return

    total_bars = 0
    annotated = 0

    base = Path(dataset_dir)
    for i, img_file in enumerate(image_files, 1):
        rel = img_file.relative_to(base)
        print(f"[{i}/{len(image_files)}] {rel}")

        bars = detect_hp_bars(str(img_file))

        if bars:
            xml_path, new_count = _save_voc(str(img_file), bars)
            print(f"  检测到 {len(bars)} 条血条，新增 {new_count} 条标注 -> {Path(xml_path).name}")
            total_bars += len(bars)
            annotated += 1
        else:
            print("  未检测到血条")

    print(f"\n处理完成: {annotated}/{len(image_files)} 张图片有血条，共 {total_bars} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="检测敌人血条并生成 VOC 标注")
    parser.add_argument("--dataset_dir", required=True, help="数据集目录")
    args = parser.parse_args()
    run(args.dataset_dir)


if __name__ == "__main__":
    main()
