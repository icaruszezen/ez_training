"""预标注引擎，协调图片处理、模型调用和标注文件生成"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from ez_traing.prelabeling.models import (
    BoundingBox,
    DetectionMode,
    DetectionResult,
    InferenceBackend,
    PrelabelingStats,
)
from ez_traing.prelabeling.vision_service import VisionModelService
from ez_traing.prelabeling.voc_writer import VOCAnnotationWriter
from ez_traing.prelabeling.yolo_service import YoloModelService

logger = logging.getLogger(__name__)


def validate_prelabeling_input(
    prompt: str,
    config_manager,
    inference_backend: str = "vision_api",
    yolo_model_path: str = "",
    detection_mode: str = "text_only",
    reference_images: List[str] = None,
) -> bool:
    """验证预标注输入参数

    在启动工作线程前调用，以便 UI 层能立即显示错误信息。

    Args:
        prompt: 用户输入的提示词
        config_manager: APIConfigManager 实例
        detection_mode: 检测模式，"text_only" 或 "reference_image"
        reference_images: 参考图片路径列表（参考图片模式下必须提供）

    Returns:
        True 如果所有验证通过

    Raises:
        ValueError: 文本模式下提示词为空或仅包含空白字符
        ValueError: 参考图片模式下未提供参考图片
        ValueError: API 配置不完整（endpoint 或 api_key 为空）
    """
    backend = InferenceBackend(inference_backend)

    if backend == InferenceBackend.YOLO_PT:
        model_path = (yolo_model_path or "").strip()
        if not model_path:
            raise ValueError("请先选择 YOLO 权重文件（.pt）")
        path = Path(model_path)
        if not path.exists() or not path.is_file():
            raise ValueError("YOLO 权重文件不存在")
        if path.suffix.lower() != ".pt":
            raise ValueError("YOLO 权重文件必须是 .pt 格式")
        return True

    is_reference_mode = detection_mode == DetectionMode.REFERENCE_IMAGE.value

    if is_reference_mode:
        if not reference_images:
            raise ValueError("参考图片模式下必须提供至少一张参考图片")
    else:
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空")

    if not config_manager.is_configured():
        raise ValueError("API 配置不完整，请先设置 API 地址和令牌")

    return True


class PrelabelingWorker(QThread):
    """预标注工作线程

    在后台线程中批量处理图片列表，调用视觉模型进行目标检测，
    并将结果保存为 VOC 格式标注文件。

    Signals:
        progress(int, int, str): 当前进度 (current, total, message)
        image_completed(str, bool, str): 单张图片完成 (path, success, message)
        finished(object): 批量处理完成，携带 PrelabelingStats 统计
    """

    progress = pyqtSignal(int, int, str)
    image_completed = pyqtSignal(str, bool, str)
    finished = pyqtSignal(object)

    def __init__(
        self,
        image_paths: List[str],
        prompt: str,
        vision_service: Optional[VisionModelService] = None,
        yolo_service: Optional[YoloModelService] = None,
        inference_backend: str = "vision_api",
        skip_annotated: bool = True,
        overwrite: bool = False,
        max_workers: int = 1,
        reference_images: List[str] = None,
        detection_mode: str = "text_only",
    ):
        super().__init__()
        self._image_paths = image_paths
        self._prompt = prompt
        self._vision_service = vision_service
        self._yolo_service = yolo_service
        self._inference_backend = InferenceBackend(inference_backend)
        self._skip_annotated = skip_annotated
        self._overwrite = overwrite
        self._is_cancelled = False
        self._voc_writer = VOCAnnotationWriter()
        self._max_workers = max(1, max_workers)

        # 验证并存储检测模式
        self._detection_mode = DetectionMode(detection_mode)

        # 验证参考图片参数
        if (
            self._inference_backend == InferenceBackend.VISION_API
            and self._detection_mode == DetectionMode.REFERENCE_IMAGE
        ):
            if not reference_images:
                raise ValueError("参考图片模式下必须提供至少一张参考图片")

        if self._inference_backend == InferenceBackend.VISION_API and self._vision_service is None:
            raise ValueError("视觉 API 模式下必须提供 vision_service")
        if self._inference_backend == InferenceBackend.YOLO_PT and self._yolo_service is None:
            raise ValueError("YOLO 模式下必须提供 yolo_service")

        self._reference_images = reference_images or []


    def run(self) -> None:
        """执行预标注处理

        使用线程池并发调用视觉模型检测目标，并将结果保存为 VOC 标注文件。
        支持跳过已标注图片和中途取消。
        """
        stats = PrelabelingStats(total=len(self._image_paths))

        if self._max_workers <= 1:
            self._run_sequential(stats)
        else:
            self._run_concurrent(stats)

        self.finished.emit(stats)

    # ------------------------------------------------------------------
    # Sequential (single-thread) path
    # ------------------------------------------------------------------

    def _run_sequential(self, stats: PrelabelingStats) -> None:
        for i, image_path in enumerate(self._image_paths):
            if self._is_cancelled:
                logger.info("预标注已被用户取消")
                break
            self._process_one(image_path, i, stats)

    # ------------------------------------------------------------------
    # Concurrent (multi-thread) path
    # ------------------------------------------------------------------

    def _run_concurrent(self, stats: PrelabelingStats) -> None:
        lock = threading.Lock()
        completed_count = 0

        def _worker(image_path: str, index: int):
            nonlocal completed_count
            if self._is_cancelled:
                return
            self._process_one(image_path, index, stats, lock)
            with lock:
                completed_count += 1

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(_worker, path, i): path
                for i, path in enumerate(self._image_paths)
            }
            for future in as_completed(futures):
                if self._is_cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                # 让异常传播到日志而不是静默吞掉
                try:
                    future.result()
                except Exception:
                    logger.exception("处理图片时发生未预期的异常")

    # ------------------------------------------------------------------
    # Single image processing (thread-safe)
    # ------------------------------------------------------------------

    def _process_one(self, image_path: str, index: int,
                     stats: PrelabelingStats,
                     lock: threading.Lock = None) -> None:
        """处理单张图片，线程安全"""
        if self._is_cancelled:
            return

        def _update_stats(attr: str):
            if lock:
                with lock:
                    setattr(stats, attr, getattr(stats, attr) + 1)
            else:
                setattr(stats, attr, getattr(stats, attr) + 1)

        # 跳过已标注图片
        if self._skip_annotated and not self._overwrite and self._has_annotation(image_path):
            _update_stats("skipped")
            msg = f"已有标注，跳过: {Path(image_path).name}"
            logger.info(msg)
            self.progress.emit(index + 1, stats.total, msg)
            self.image_completed.emit(image_path, True, msg)
            return

        # 调用视觉模型检测
        filename = Path(image_path).name
        self.progress.emit(index + 1, stats.total, f"正在处理: {filename}")

        if self._inference_backend == InferenceBackend.YOLO_PT:
            result: DetectionResult = self._yolo_service.detect_objects(image_path)
        else:
            if self._detection_mode == DetectionMode.REFERENCE_IMAGE:
                result = self._vision_service.detect_objects_with_reference(
                    self._reference_images, image_path, self._prompt
                )
            else:
                result = self._vision_service.detect_objects(image_path, self._prompt)

        if not result.success:
            _update_stats("processed")
            _update_stats("failed")
            msg = f"检测失败: {result.error_message}"
            logger.error("图片 %s %s", filename, msg)
            self.image_completed.emit(image_path, False, msg)
            return

        if not result.boxes:
            _update_stats("processed")
            _update_stats("success")
            msg = f"未检测到目标: {filename}"
            logger.info(msg)
            self.image_completed.emit(image_path, True, msg)
            return

        # 保存 VOC 标注
        try:
            should_merge = (not self._skip_annotated) and self._has_annotation(image_path)
            self._save_voc_annotation(
                image_path,
                result.boxes,
                merge_existing=should_merge,
            )
            _update_stats("processed")
            _update_stats("success")
            if should_merge:
                msg = f"标注合并完成: {filename}（新增 {len(result.boxes)} 个目标）"
            else:
                msg = f"标注完成: {filename} ({len(result.boxes)} 个目标)"
            logger.info(msg)
            self.image_completed.emit(image_path, True, msg)
        except Exception as e:
            _update_stats("processed")
            _update_stats("failed")
            msg = f"保存标注失败: {e}"
            logger.error("图片 %s %s", filename, msg)
            self.image_completed.emit(image_path, False, msg)

    def cancel(self) -> None:
        """取消批量处理"""
        self._is_cancelled = True

    def _has_annotation(self, image_path: str) -> bool:
        """检查图片是否已有 VOC 标注文件

        Args:
            image_path: 图片文件路径

        Returns:
            True 如果同目录下存在同名 .xml 文件
        """
        xml_path = Path(image_path).with_suffix(".xml")
        return xml_path.exists()

    def _save_voc_annotation(
        self,
        image_path: str,
        boxes: List[BoundingBox],
        merge_existing: bool = False,
    ) -> str:
        """保存 VOC 格式标注文件

        Args:
            image_path: 图片文件路径
            boxes: 检测到的边界框列表

        Returns:
            保存的标注文件路径
        """
        image_size = self._voc_writer._get_image_size(image_path)
        if merge_existing:
            return self._voc_writer.save_merged_annotation(image_path, image_size, boxes)
        return self._voc_writer.save_annotation(image_path, image_size, boxes)


