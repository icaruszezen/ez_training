"""视觉模型预标注模块"""

from ez_traing.prelabeling.config import APIConfigManager
from ez_traing.prelabeling.engine import PrelabelingWorker, validate_prelabeling_input
from ez_traing.prelabeling.models import (
    BoundingBox,
    DetectionResult,
    InferenceBackend,
    PrelabelingStats,
    VisionAPIConfig,
)
from ez_traing.prelabeling.vision_service import VisionModelService
from ez_traing.prelabeling.voc_writer import VOCAnnotationWriter
from ez_traing.prelabeling.yolo_service import YoloModelService

__all__ = [
    "APIConfigManager",
    "BoundingBox",
    "DetectionResult",
    "InferenceBackend",
    "PrelabelingStats",
    "PrelabelingWorker",
    "VisionAPIConfig",
    "VisionModelService",
    "VOCAnnotationWriter",
    "YoloModelService",
    "validate_prelabeling_input",
]
