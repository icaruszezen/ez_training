"""本地 YOLO .pt 推理服务。"""

import gc
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ez_training.prelabeling.models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


def _is_oom_error(exc: Exception) -> bool:
    """判断异常是否为 GPU OOM。"""
    cls_name = type(exc).__name__
    if cls_name == "OutOfMemoryError":
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


class YoloModelService:
    """基于 Ultralytics YOLO 的本地推理服务。

    内置类级别模型缓存：相同 model_path 只加载一次。
    """

    _model_cache: Dict[str, object] = {}
    _cache_lock = threading.Lock()

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

        resolved = str(path.resolve())
        with self._cache_lock:
            if resolved in self._model_cache:
                logger.info("复用已缓存的 YOLO 模型: %s", self._model_path)
                self._model = self._model_cache[resolved]
            else:
                self._evict_cache_locked()
                self._model = YOLO(self._model_path)
                self._model_cache[resolved] = self._model
                logger.info("YOLO 模型已加载并缓存: %s", self._model_path)

    @classmethod
    def _evict_cache_locked(cls) -> None:
        """Evict all cached models to free GPU memory. Caller must hold _cache_lock."""
        if not cls._model_cache:
            return
        cls._model_cache.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("已清理 YOLO 模型缓存并释放显存")

    @classmethod
    def clear_model_cache(cls) -> None:
        """清理模型缓存并释放 GPU 显存。"""
        with cls._cache_lock:
            cls._evict_cache_locked()

    def _predict(
        self, image_path: str, device: Optional[str] = None,
    ) -> list:
        return self._model.predict(
            source=image_path,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=device if device is not None else self._device,
            verbose=False,
        )

    def detect_objects(self, image_path: str) -> DetectionResult:
        """对单张图片执行本地 YOLO 推理。GPU OOM 时自动降级到 CPU 重试。"""
        try:
            results = self._predict(image_path)
        except Exception as e:
            if _is_oom_error(e) and self._device != "cpu":
                logger.warning("GPU 显存不足，自动降级到 CPU 重试: %s", e)
                try:
                    results = self._predict(image_path, device="cpu")
                except Exception as cpu_e:
                    msg = f"YOLO CPU 降级推理也失败: {cpu_e}"
                    logger.exception(msg)
                    return DetectionResult(success=False, error_message=msg)
            else:
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
