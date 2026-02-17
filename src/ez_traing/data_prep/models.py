"""数据准备流程数据模型。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AnnotationBox:
    """VOC/YOLO 统一框结构。"""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class DatasetSample:
    """单张样本数据。"""

    image_path: Path
    xml_path: Optional[Path]
    boxes: List[AnnotationBox] = field(default_factory=list)


@dataclass
class DataPrepConfig:
    """数据准备配置。"""

    dataset_name: str
    dataset_dir: str
    output_dir: str
    train_ratio: float = 0.8
    random_seed: int = 42
    augment_methods: List[str] = field(default_factory=list)
    augment_times: int = 1
    augment_workers: int = 0  # 0 表示自动按 CPU 与增强次数决定
    augment_scope: str = "train"  # train | both
    skip_unlabeled: bool = True
    overwrite_output: bool = True

    def validate(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("数据集名称不能为空")
        if not self.dataset_dir.strip():
            raise ValueError("数据集目录不能为空")
        if not self.output_dir.strip():
            raise ValueError("输出目录不能为空")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("训练集比例必须在 0 到 1 之间")
        if self.augment_scope not in {"train", "both"}:
            raise ValueError("增强范围必须是 train 或 both")
        if self.augment_times < 0:
            raise ValueError("增强次数不能为负数")
        if self.augment_workers < 0:
            raise ValueError("增强线程数不能为负数")


@dataclass
class DataPrepSummary:
    """数据准备结果摘要。"""

    dataset_name: str
    output_dir: str
    source_images: int = 0
    processed_images: int = 0
    train_images: int = 0
    val_images: int = 0
    augmented_images: int = 0
    skipped_images: int = 0
    classes_count: int = 0
    yaml_path: str = ""
    classes_path: str = ""
