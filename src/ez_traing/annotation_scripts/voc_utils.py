"""VOC 标注格式公共工具函数。

提供 VOC XML 的创建 / 读取 / 合并，以及通用的"扫描目录 -> 检测 -> 保存标注"
运行骨架，供各预标注脚本复用。
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Tuple

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS

DetectResult = Tuple[List[Tuple[int, int, int, int]], int, int]


# ---------------------------------------------------------------------------
# VOC XML 读写
# ---------------------------------------------------------------------------

def create_voc_root(
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


def append_object(
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


def read_existing_objects(
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


def save_voc(
    image_path: str,
    label: str,
    bars: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
    depth: int = 3,
) -> Tuple[str, int]:
    """保存或合并 VOC 标注，返回 (xml_path, 新增框数)。"""
    img_path = Path(image_path)
    xml_path = str(img_path.with_suffix(".xml"))

    folder = img_path.parent.name
    filename = img_path.name

    existing_objects = read_existing_objects(xml_path)

    root = create_voc_root(folder, filename, str(img_path), width, height, depth)

    added_keys: set = set()
    for name, xmin, ymin, xmax, ymax in existing_objects:
        key = (name, xmin, ymin, xmax, ymax)
        if key not in added_keys:
            append_object(root, name, xmin, ymin, xmax, ymax)
            added_keys.add(key)

    new_count = 0
    for xmin, ymin, xmax, ymax in bars:
        key = (label, xmin, ymin, xmax, ymax)
        if key not in added_keys:
            append_object(root, label, xmin, ymin, xmax, ymax)
            added_keys.add(key)
            new_count += 1

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    xml_dir = str(Path(xml_path).parent)
    fd, tmp = tempfile.mkstemp(dir=xml_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        Path(tmp).replace(xml_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    return xml_path, new_count


# ---------------------------------------------------------------------------
# 通用预标注运行骨架
# ---------------------------------------------------------------------------

def run_annotation(
    dataset_dir: str,
    label: str,
    detect_fn: Callable[[str], DetectResult],
    item_name: str = "目标",
) -> None:
    """扫描目录下所有图片，调用 *detect_fn* 检测并保存 VOC 标注。

    Args:
        dataset_dir: 数据集根目录。
        label: 写入 VOC ``<name>`` 的类别标签。
        detect_fn: ``(image_path) -> (bars, width, height)``。
        item_name: 用于日志中的可读名称（如"横条""血条"）。
    """
    print(f"开始处理数据集: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        print(f"错误: 目录不存在 - {dataset_dir}")
        return

    image_files = sorted(
        f for f in Path(dataset_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    )

    print(f"找到 {len(image_files)} 张图片（含子目录）")
    if not image_files:
        return

    total = 0
    annotated = 0

    failed = 0
    base = Path(dataset_dir)
    for i, img_file in enumerate(image_files, 1):
        rel = img_file.relative_to(base)
        print(f"[{i}/{len(image_files)}] {rel}")

        try:
            bars, width, height = detect_fn(str(img_file))
        except Exception as e:
            print(f"  处理失败: {e}")
            failed += 1
            continue

        if bars:
            xml_path, new_count = save_voc(str(img_file), label, bars, width, height)
            print(
                f"  检测到 {len(bars)} 条{item_name}，"
                f"新增 {new_count} 条标注 -> {Path(xml_path).name}"
            )
            total += len(bars)
            annotated += 1
        else:
            print(f"  未检测到{item_name}")

    summary = f"\n处理完成: {annotated}/{len(image_files)} 张图片有{item_name}，共 {total} 条"
    if failed:
        summary += f"，{failed} 张处理失败"
    print(summary)
