"""自动检测游戏截图中左下角"角色连携可用"白色横条并生成 VOC 标注。

算法流程:
1. 裁剪图片左下角 ROI 区域
2. 灰度化 + 阈值 220 二值化，提取高亮白色像素
3. 查找外轮廓，按宽高比和尺寸过滤出横条形状
4. 以 Y 中心中位数做同水平线聚合，剔除离群噪声
5. 输出 / 合并 VOC 标注
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

LABEL = "角色连携可用"

ROI_X_RATIO = 0.40
ROI_Y_RATIO = 0.25
THRESHOLD = 220

REF_IMAGE_WIDTH = 1024
REF_BAR_WIDTH = 44
BAR_WIDTH_TOLERANCE = 10
MAX_HEIGHT = 20
MIN_ASPECT_RATIO = 3.0
MAX_BARS = 4
Y_TOLERANCE = 15


# ---------------------------------------------------------------------------
# 白色横条检测
# ---------------------------------------------------------------------------

def detect_white_bars(image_path: str) -> List[Tuple[int, int, int, int]]:
    """检测图片左下角的白色横条。

    Returns:
        [(xmin, ymin, xmax, ymax), ...] 原图坐标系下的外接矩形列表。
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  无法读取图片: {image_path}")
        return []

    h, w = img.shape[:2]

    roi_x_end = int(w * ROI_X_RATIO)
    roi_y_start = int(h * (1 - ROI_Y_RATIO))
    roi = img[roi_y_start:h, 0:roi_x_end]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale = w / REF_IMAGE_WIDTH
    min_width = int((REF_BAR_WIDTH - BAR_WIDTH_TOLERANCE) * scale)
    max_width = int((REF_BAR_WIDTH + BAR_WIDTH_TOLERANCE) * scale)

    candidates: List[Tuple[int, int, int, int, float]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0 or cw == 0:
            continue
        if cw / ch < MIN_ASPECT_RATIO:
            continue
        if cw < min_width or cw > max_width:
            continue
        if ch > MAX_HEIGHT:
            continue

        orig_x = x
        orig_y = y + roi_y_start
        y_center = orig_y + ch / 2.0
        candidates.append((orig_x, orig_y, orig_x + cw, orig_y + ch, y_center))

    if not candidates:
        return []

    y_centers = [c[4] for c in candidates]
    median_y = float(np.median(y_centers))
    bars = [
        (c[0], c[1], c[2], c[3])
        for c in candidates
        if abs(c[4] - median_y) <= Y_TOLERANCE
    ]

    bars.sort(key=lambda b: b[0])

    if len(bars) > MAX_BARS:
        bars = bars[:MAX_BARS]

    return bars


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

        bars = detect_white_bars(str(img_file))

        if bars:
            xml_path, new_count = _save_voc(str(img_file), bars)
            print(f"  检测到 {len(bars)} 条横条，新增 {new_count} 条标注 -> {Path(xml_path).name}")
            total_bars += len(bars)
            annotated += 1
        else:
            print("  未检测到横条")

    print(f"\n处理完成: {annotated}/{len(image_files)} 张图片有横条，共 {total_bars} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="检测角色连携可用白色横条并生成 VOC 标注")
    parser.add_argument("--dataset_dir", required=True, help="数据集目录")
    args = parser.parse_args()
    run(args.dataset_dir)


if __name__ == "__main__":
    main()
