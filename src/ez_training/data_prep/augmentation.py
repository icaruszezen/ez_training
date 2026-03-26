"""Albumentations 数据增强封装。"""

import inspect
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ez_training.data_prep.models import AnnotationBox

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


AUGMENTATION_SPECS: List[Tuple[str, str]] = [
    ("hflip", "水平翻转 (HorizontalFlip)"),
    ("vflip", "垂直翻转 (VerticalFlip)"),
    ("rotate", "随机旋转 (Rotate)"),
    ("shift_scale_rotate", "平移缩放旋转 (ShiftScaleRotate)"),
    ("affine", "仿射变换 (Affine)"),
    ("perspective", "透视变换 (Perspective)"),
    ("random_resized_crop", "随机缩放裁剪 (RandomResizedCrop)"),
    ("brightness_contrast", "亮度对比度 (RandomBrightnessContrast)"),
    ("hsv", "色相饱和度明度 (HueSaturationValue)"),
    ("rgb_shift", "RGB 通道偏移 (RGBShift)"),
    ("clahe", "自适应直方图均衡 (CLAHE)"),
    ("gamma", "Gamma 变换 (RandomGamma)"),
    ("gaussian_blur", "高斯模糊 (GaussianBlur)"),
    ("motion_blur", "运动模糊 (MotionBlur)"),
    ("gauss_noise", "高斯噪声 (GaussNoise)"),
    ("median_blur", "中值模糊 (MedianBlur)"),
    ("coarse_dropout", "随机遮挡 (CoarseDropout)"),
]


_GEOMETRIC_METHODS = {"rotate", "shift_scale_rotate", "affine", "perspective", "random_resized_crop"}


def get_augmentation_specs() -> List[Tuple[str, str]]:
    return list(AUGMENTATION_SPECS)


def is_albumentations_available() -> bool:
    """检查 albumentations 是否已安装可用。"""
    return A is not None


def _build_transform_map(
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Callable[[], "A.BasicTransform"]]:
    if A is None:
        return {}

    return {
        "hflip": lambda: A.HorizontalFlip(p=0.6),
        "vflip": lambda: A.VerticalFlip(p=0.2),
        "rotate": lambda: A.Rotate(limit=20, p=0.5),
        "shift_scale_rotate": lambda: A.ShiftScaleRotate(
            shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.5
        ),
        "affine": lambda: A.Affine(scale=(0.9, 1.1), translate_percent=(-0.08, 0.08), p=0.4),
        "perspective": lambda: A.Perspective(scale=(0.03, 0.08), p=0.3),
        "random_resized_crop": lambda: _build_random_resized_crop(image_size),
        "brightness_contrast": lambda: A.RandomBrightnessContrast(p=0.5),
        "hsv": lambda: A.HueSaturationValue(p=0.4),
        "rgb_shift": lambda: A.RGBShift(p=0.3),
        "clahe": lambda: A.CLAHE(p=0.3),
        "gamma": lambda: A.RandomGamma(p=0.3),
        "gaussian_blur": lambda: A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        "motion_blur": lambda: A.MotionBlur(blur_limit=(3, 7), p=0.2),
        "gauss_noise": lambda: A.GaussNoise(p=0.25),
        "median_blur": lambda: A.MedianBlur(blur_limit=5, p=0.2),
        "coarse_dropout": _build_coarse_dropout,
    }


def _build_random_resized_crop(image_size: Optional[Tuple[int, int]] = None):
    """兼容 Albumentations 1.x/2.x，输出尺寸匹配原始图片。"""
    if A is None:
        raise RuntimeError("未安装 albumentations，请先安装依赖")

    h, w = image_size if image_size else (640, 640)
    params = inspect.signature(A.RandomResizedCrop).parameters
    if "size" in params:
        return A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0), ratio=(0.8, 1.25), p=0.3)
    return A.RandomResizedCrop(height=h, width=w, scale=(0.7, 1.0), ratio=(0.8, 1.25), p=0.3)


def _build_coarse_dropout():
    """兼容 Albumentations 1.x/2.x 的 CoarseDropout 参数差异。"""
    if A is None:
        raise RuntimeError("未安装 albumentations，请先安装依赖")

    params = inspect.signature(A.CoarseDropout).parameters
    if "num_holes_range" in params:
        return A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(0.02, 0.06),
            hole_width_range=(0.02, 0.06),
            p=0.25,
        )
    return A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.25)


def build_augmenter(
    methods: List[str], image_size: Optional[Tuple[int, int]] = None
):
    """根据方法列表构建增强器。

    几何类变换（rotate / shift_scale_rotate / affine / perspective /
    random_resized_crop）使用 ``OneOf`` 随机选取一种执行，避免多种几何变换
    叠加导致图像过度失真。翻转、颜色、模糊、噪声等变换各自独立应用。

    Args:
        methods: 增强方法名称列表。
        image_size: ``(height, width)``，用于 RandomResizedCrop 等尺寸相关变换。
    """
    if not methods:
        return None
    if A is None:
        raise RuntimeError("未安装 albumentations，请先安装依赖")

    transform_map = _build_transform_map(image_size=image_size)

    geometric_transforms = []
    other_transforms = []
    for m in methods:
        if m not in transform_map:
            continue
        t = transform_map[m]()
        if m in _GEOMETRIC_METHODS:
            geometric_transforms.append(t)
        else:
            other_transforms.append(t)

    final_transforms = list(other_transforms)
    if geometric_transforms:
        if len(geometric_transforms) == 1:
            final_transforms.append(geometric_transforms[0])
        else:
            final_transforms.append(A.OneOf(geometric_transforms, p=0.6))

    if not final_transforms:
        return None

    return A.Compose(
        final_transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.1,
        ),
        p=1.0,
    )


def apply_augmentation(
    image: np.ndarray, boxes: List[AnnotationBox], augmenter
) -> Tuple[np.ndarray, List[AnnotationBox]]:
    """对图像和框执行一次增强。"""
    if augmenter is None:
        return image, list(boxes)

    bboxes = [[b.x_min, b.y_min, b.x_max, b.y_max] for b in boxes]
    labels = [b.label for b in boxes]

    result = augmenter(image=image, bboxes=bboxes, class_labels=labels)
    aug_image = result["image"]
    aug_boxes_raw = result["bboxes"]
    aug_labels = result["class_labels"]

    h, w = aug_image.shape[:2]
    aug_boxes: List[AnnotationBox] = []
    for label, box in zip(aug_labels, aug_boxes_raw):
        x_min, y_min, x_max, y_max = box
        x_min = max(0.0, min(float(x_min), float(w)))
        x_max = max(0.0, min(float(x_max), float(w)))
        y_min = max(0.0, min(float(y_min), float(h)))
        y_max = max(0.0, min(float(y_max), float(h)))
        if x_max <= x_min or y_max <= y_min:
            continue
        aug_boxes.append(
            AnnotationBox(
                label=str(label),
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )

    return aug_image, aug_boxes
