"""本地 YOLO .pt 推理服务。"""

import logging
from pathlib import Path
from typing import List, Optional

from ez_traing.prelabeling.models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


class YoloModelService:
    """基于 Ultralytics YOLO 的本地推理服务。"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self._model_path = str(model_path).strip()
        self._conf_threshold = float(conf_threshold)
        self._iou_threshold = float(iou_threshold)
        self._device = device

        if not self._model_path:
            raise ValueError("请先选择 YOLO 权重文件（.pt）")

        path = Path(self._model_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"YOLO 权重文件不存在: {self._model_path}")
        if path.suffix.lower() != ".pt":
            raise ValueError("YOLO 权重文件必须是 .pt 格式")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "未安装 ultralytics，请先安装后再使用本地 YOLO 推理"
            ) from e

        self._model = YOLO(self._model_path)

    def detect_objects(self, image_path: str) -> DetectionResult:
        """对单张图片执行本地 YOLO 推理。"""
        try:
            results = self._model.predict(
                source=image_path,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            msg = f"YOLO 推理失败: {e}"
            logger.exception(msg)
            return DetectionResult(success=False, error_message=msg)

        if not results:
            return DetectionResult(success=True, boxes=[])

        result = results[0]
        boxes_data = result.boxes
        names = result.names if hasattr(result, "names") else {}
        boxes: List[BoundingBox] = []

        if boxes_data is None:
            return DetectionResult(success=True, boxes=[])

        for item in boxes_data:
            try:
                x_min, y_min, x_max, y_max = [int(round(v)) for v in item.xyxy[0].tolist()]
                cls_id = int(item.cls.item()) if item.cls is not None else -1
                conf = float(item.conf.item()) if item.conf is not None else 1.0
            except Exception:
                continue

            if isinstance(names, dict):
                label = str(names.get(cls_id, cls_id))
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                label = str(names[cls_id])
            else:
                label = str(cls_id)

            boxes.append(
                BoundingBox(
                    label=label,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=conf,
                )
            )

        return DetectionResult(success=True, boxes=boxes)
