"""VOC 标注格式公共工具函数。

提供 VOC XML 的创建 / 读取 / 合并，以及通用的"扫描目录 -> 检测 -> 保存标注"
运行骨架，供各预标注脚本复用。
"""

import logging
import os
from pathlib import Path
from typing import Callable, List, Tuple

from ez_training.common.constants import SUPPORTED_IMAGE_FORMATS
from ez_training.common.voc_io import (
    VocObject,
    append_voc_object,
    create_voc_xml,
    parse_voc_objects,
    save_voc_xml,
)

logger = logging.getLogger(__name__)

DetectResult = Tuple[List[Tuple[int, int, int, int]], int, int]

# Re-export for backward compatibility
create_voc_root = create_voc_xml


def append_object(
    root, label: str, xmin: int, ymin: int, xmax: int, ymax: int,
) -> None:
    """Backward-compatible wrapper around :func:`append_voc_object`."""
    append_voc_object(root, label, xmin, ymin, xmax, ymax)


def read_existing_objects(
    xml_path: str,
) -> List[Tuple[str, int, int, int, int]]:
    """读取已有 VOC XML 中的所有目标框。"""
    return [
        (o.label, o.xmin, o.ymin, o.xmax, o.ymax)
        for o in parse_voc_objects(xml_path)
    ]


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

    root = create_voc_xml(folder, filename, str(img_path), width, height, depth)

    added_keys: set = set()
    for name, xmin, ymin, xmax, ymax in existing_objects:
        key = (name, xmin, ymin, xmax, ymax)
        if key not in added_keys:
            append_voc_object(root, name, xmin, ymin, xmax, ymax)
            added_keys.add(key)

    new_count = 0
    for xmin, ymin, xmax, ymax in bars:
        key = (label, xmin, ymin, xmax, ymax)
        if key not in added_keys:
            append_voc_object(root, label, xmin, ymin, xmax, ymax)
            added_keys.add(key)
            new_count += 1

    save_voc_xml(root, xml_path)
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
