"""视觉模型预标注功能的数据模型定义"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class DetectionMode(Enum):
    """检测模式枚举"""
    TEXT_ONLY = "text_only"
    REFERENCE_IMAGE = "reference_image"


class InferenceBackend(Enum):
    """推理后端枚举"""

    VISION_API = "vision_api"
    YOLO_PT = "yolo_pt"


@dataclass
class ReferenceImageInfo:
    """参考图片信息"""
    path: str
    is_valid: bool = True
    error_message: str = ""


@dataclass
class VisionAPIConfig:
    """视觉模型 API 配置"""
    endpoint: str = ""
    api_key: str = ""
    model_name: str = "gpt-4-vision-preview"
    timeout: int = 60


@dataclass
class BoundingBox:
    """检测到的边界框"""
    label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0


@dataclass
class DetectionResult:
    """检测结果"""
    success: bool
    boxes: List[BoundingBox] = field(default_factory=list)
    error_message: str = ""
    raw_response: str = ""


@dataclass
class PrelabelingStats:
    """预标注统计"""
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
