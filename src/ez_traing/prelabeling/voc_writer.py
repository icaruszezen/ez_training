"""VOC 标注文件写入器，封装 PascalVocWriter 提供简化接口"""

import sys
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from ez_traing.prelabeling.models import BoundingBox

# 确保 labelImg 的 libs 模块可被导入
_LABELIMG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "labelImg"
if str(_LABELIMG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LABELIMG_ROOT))

from libs.pascal_voc_io import PascalVocWriter


class VOCAnnotationWriter:
    """VOC 标注文件写入器"""

    @staticmethod
    def _deduplicate_boxes(boxes: List[BoundingBox]) -> List[BoundingBox]:
        """按标签和坐标去重，保留出现顺序。"""
        seen = set()
        merged: List[BoundingBox] = []
        for box in boxes:
            key = (box.label, box.x_min, box.y_min, box.x_max, box.y_max)
            if key in seen:
                continue
            seen.add(key)
            merged.append(box)
        return merged

    def save_annotation(
        self,
        image_path: str,
        image_size: Tuple[int, int, int],
        boxes: List[BoundingBox],
        output_path: Optional[str] = None,
    ) -> str:
        """
        保存 VOC 格式标注文件

        Args:
            image_path: 图片路径
            image_size: 图片尺寸 (height, width, depth)
            boxes: 边界框列表
            output_path: 输出路径，默认与图片同目录同名 .xml

        Returns:
            保存的文件路径
        """
        img_path = Path(image_path)
        folder_name = img_path.parent.name
        filename = img_path.name

        if output_path is None:
            output_path = str(img_path.with_suffix(".xml"))

        writer = PascalVocWriter(
            folder_name,
            filename,
            image_size,
            database_src="Unknown",
            local_img_path=str(img_path),
        )

        for box in boxes:
            writer.add_bnd_box(
                box.x_min, box.y_min, box.x_max, box.y_max, box.label, difficult=0
            )

        writer.save(output_path)
        return output_path

    def read_annotation(self, xml_path: str) -> List[BoundingBox]:
        """读取已有 VOC 标注并转换为 BoundingBox 列表。"""
        path = Path(xml_path)
        if not path.exists():
            return []

        root = ET.parse(path).getroot()
        boxes: List[BoundingBox] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if not name or bnd is None:
                continue
            try:
                x_min = int(float((bnd.findtext("xmin") or "0").strip()))
                y_min = int(float((bnd.findtext("ymin") or "0").strip()))
                x_max = int(float((bnd.findtext("xmax") or "0").strip()))
                y_max = int(float((bnd.findtext("ymax") or "0").strip()))
            except ValueError:
                continue

            boxes.append(
                BoundingBox(
                    label=name,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=1.0,
                )
            )
        return boxes

    def save_merged_annotation(
        self,
        image_path: str,
        image_size: Tuple[int, int, int],
        boxes: List[BoundingBox],
        output_path: Optional[str] = None,
    ) -> str:
        """将识别结果与已有 VOC 标注合并后保存。"""
        img_path = Path(image_path)
        xml_path = Path(output_path) if output_path else img_path.with_suffix(".xml")
        existing_boxes = self.read_annotation(str(xml_path))
        all_boxes = self._deduplicate_boxes(existing_boxes + boxes)
        return self.save_annotation(
            image_path=image_path,
            image_size=image_size,
            boxes=all_boxes,
            output_path=str(xml_path),
        )

    def _get_image_size(self, image_path: str) -> Tuple[int, int, int]:
        """
        获取图片尺寸

        Args:
            image_path: 图片文件路径

        Returns:
            (height, width, depth) 元组
        """
        with Image.open(image_path) as img:
            width, height = img.size
            # RGB -> 3, L (grayscale) -> 1, RGBA -> 4, etc.
            mode_to_depth = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
            depth = mode_to_depth.get(img.mode, 3)
        return (height, width, depth)
