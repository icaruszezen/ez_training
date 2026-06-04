"""训练前数据准备数据模型。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

EXPORT_FORMAT_YOLO = "yolo"
EXPORT_FORMAT_VOC = "voc"
VALID_EXPORT_FORMATS = {
    EXPORT_FORMAT_YOLO,
    EXPORT_FORMAT_VOC,
}

IMAGE_EXPORT_RULE_EXCLUDE_IF_ANY_UNSELECTED = "exclude_if_any_unselected"
IMAGE_EXPORT_RULE_INCLUDE_IF_ANY_SELECTED = "include_if_any_selected"
VALID_IMAGE_EXPORT_RULES = {
    IMAGE_EXPORT_RULE_EXCLUDE_IF_ANY_UNSELECTED,
    IMAGE_EXPORT_RULE_INCLUDE_IF_ANY_SELECTED,
}


def load_custom_class_names(custom_classes_file: Union[str, Path]) -> List[str]:
    """读取自定义 classes.txt，保留原始顺序并忽略空行。"""
    path = Path(custom_classes_file)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_mode: Optional[str] = None


@dataclass
class DataPrepConfig:
    """数据准备配置。"""

    dataset_name: str
    dataset_dir: str
    output_dir: str
    export_format: str = EXPORT_FORMAT_YOLO
    train_ratio: float = 0.8
    random_seed: int = 42
    augment_methods: List[str] = field(default_factory=list)
    augment_times: int = 1
    augment_workers: int = 0  # 0 表示自动按 CPU 与增强次数决定
    augment_scope: str = "train"  # train | both
    skip_unlabeled: bool = True
    overwrite_output: bool = True
    custom_classes_file: Optional[str] = None
    selected_classes: List[str] = field(default_factory=list)
    image_export_rule: str = IMAGE_EXPORT_RULE_EXCLUDE_IF_ANY_UNSELECTED
    dataset_dirs: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("数据集名称不能为空")
        if not self.dataset_dirs and not self.dataset_dir.strip():
            raise ValueError("数据集目录不能为空")
        if not self.output_dir.strip():
            raise ValueError("输出目录不能为空")
        if self.export_format not in VALID_EXPORT_FORMATS:
            raise ValueError(f"不支持的输出格式: {self.export_format}")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("训练集比例必须在 0 到 1 之间")
        if self.augment_scope not in {"train", "both"}:
            raise ValueError("增强范围必须是 train 或 both")
        if self.augment_times < 0:
            raise ValueError("增强次数不能为负数")
        if self.augment_workers < 0:
            raise ValueError("增强线程数不能为负数")
        if self.image_export_rule not in VALID_IMAGE_EXPORT_RULES:
            raise ValueError(f"不支持的整图导出规则: {self.image_export_rule}")

        if self.custom_classes_file is not None:
            path = Path(self.custom_classes_file)
            if not path.exists():
                raise ValueError(f"自定义类别文件不存在: {path}")

            class_names = load_custom_class_names(path)
            if not class_names:
                raise ValueError(f"自定义类别文件为空: {path}")
            if not self.selected_classes:
                raise ValueError("请至少选择一个需要导出的类别")

            class_set = set(class_names)
            invalid_selected = [
                name for name in self.selected_classes if name not in class_set
            ]
            if invalid_selected:
                raise ValueError(
                    f"所选导出类别不在自定义类别文件中: {invalid_selected}"
                )


@dataclass
class DataPrepSummary:
    """数据准备结果摘要。"""

    dataset_name: str
    output_dir: str
    export_format: str = EXPORT_FORMAT_YOLO
    source_images: int = 0
    processed_images: int = 0
    train_images: int = 0
    val_images: int = 0
    augmented_images: int = 0
    skipped_images: int = 0
    classes_count: int = 0
    yaml_path: str = ""
    classes_path: str = ""
    train_list_path: str = ""
    val_list_path: str = ""
    trainval_list_path: str = ""
