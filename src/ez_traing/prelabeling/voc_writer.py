"""VOC 标注文件写入器，封装 PascalVocWriter 提供简化接口"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from ez_traing.common.voc_io import parse_voc_objects
from ez_traing.labeling.pascal_voc_io import PascalVocWriter
from ez_traing.prelabeling.models import BoundingBox

logger = logging.getLogger(__name__)

_CACHE_MAX_SIZE = 512


class VOCAnnotationWriter:
    """VOC 标注文件写入器"""

    _annotation_cache: OrderedDict = OrderedDict()
    _image_size_cache: OrderedDict = OrderedDict()
    _cache_lock = threading.Lock()

    @classmethod
    def _cache_put(cls, cache: OrderedDict, key, value) -> None:
        cache[key] = value
        while len(cache) > _CACHE_MAX_SIZE:
            cache.popitem(last=False)

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

        height, width = image_size[0], image_size[1]
        for box in boxes:
            x_min = max(0, min(box.x_min, width))
            y_min = max(0, min(box.y_min, height))
            x_max = max(0, min(box.x_max, width))
            y_max = max(0, min(box.y_max, height))
            if x_min >= x_max or y_min >= y_max:
                logger.warning(
                    "退化框已跳过: label=%s coords=(%d,%d,%d,%d)",
                    box.label, box.x_min, box.y_min, box.x_max, box.y_max,
                )
                continue
            writer.add_bnd_box(
                x_min, y_min, x_max, y_max, box.label, difficult=0
            )

        writer.save(output_path)
        return output_path

    def read_annotation(self, xml_path: str) -> List[BoundingBox]:
        """读取已有 VOC 标注并转换为 BoundingBox 列表。"""
        path = Path(xml_path).resolve()
        if not path.exists():
            return []

        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return []
        cache_key = (str(path), mtime_ns)

        with self._cache_lock:
            cached = self._annotation_cache.get(cache_key)
            if cached is not None:
                return list(cached)

        boxes = [
            BoundingBox(
                label=o.label,
                x_min=o.xmin,
                y_min=o.ymin,
                x_max=o.xmax,
                y_max=o.ymax,
                confidence=1.0,
            )
            for o in parse_voc_objects(path)
        ]

        with self._cache_lock:
            self._cache_put(self._annotation_cache, cache_key, boxes)
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

    def get_image_size(self, image_path: str) -> Tuple[int, int, int]:
        """
        获取图片尺寸

        Args:
            image_path: 图片文件路径

        Returns:
            (height, width, depth) 元组
        """
        path = Path(image_path).resolve()
        mtime_ns = path.stat().st_mtime_ns
        cache_key = (str(path), mtime_ns)

        with self._cache_lock:
            cached = self._image_size_cache.get(cache_key)
            if cached is not None:
                return cached

        with Image.open(path) as img:
            width, height = img.size
            # RGB -> 3, L (grayscale) -> 1, RGBA -> 4, etc.
            mode_to_depth = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
            depth = mode_to_depth.get(img.mode, 3)
        size = (height, width, depth)

        with self._cache_lock:
            self._cache_put(self._image_size_cache, cache_key, size)
        return size
