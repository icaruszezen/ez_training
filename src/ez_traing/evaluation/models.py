"""模型验证相关数据结构。"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any


@dataclass
class EvalConfig:
    """验证配置。"""

    dataset_name: str
    dataset_dir: str
    model_path: str
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    conf: float = 0.25
    iou: float = 0.45
    source: str = "custom"
    output_root: str = ""
    include_unannotated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalMetrics:
    """核心评估指标。"""

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """验证结果。"""

    success: bool
    message: str = ""
    save_dir: str = ""
    data_yaml: str = ""
    metrics: Optional[EvalMetrics] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    raw_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.metrics is not None:
            result["metrics"] = self.metrics.to_dict()
        return result
