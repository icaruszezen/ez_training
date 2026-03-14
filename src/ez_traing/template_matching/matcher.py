"""OpenCV 模板匹配引擎，封装 cv2.matchTemplate 并提供 NMS 去重。"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ez_traing.common.image_utils import imread_unicode
from ez_traing.prelabeling.models import BoundingBox

logger = logging.getLogger(__name__)

_SQDIFF_METHODS = frozenset({cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED})


@dataclass
class PreprocessConfig:
    """模板匹配的图像预处理配置。

    创建模板时由用户配置，匹配时对模板图和目标图施加相同的预处理流水线。
    执行顺序: 灰度化 -> 高斯模糊 -> 二值化/自适应二值化 -> Canny 边缘检测。
    """

    to_grayscale: bool = False
    gaussian_blur_ksize: int = 0
    binary_threshold: int = -1
    binary_inverse: bool = False
    use_adaptive_threshold: bool = False
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    canny_enabled: bool = False
    canny_low: int = 50
    canny_high: int = 150
    target_roi: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["target_roi"] is not None:
            d["target_roi"] = list(d["target_roi"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessConfig":
        roi = data.get("target_roi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            data = {**data, "target_roi": tuple(roi)}
        else:
            data = {**data, "target_roi": None}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def needs_grayscale(self) -> bool:
        """二值化和 Canny 隐式要求灰度输入。"""
        return (
            self.to_grayscale
            or self.binary_threshold >= 0
            or self.use_adaptive_threshold
            or self.canny_enabled
        )


@dataclass
class TemplateInfo:
    """单个模板图的元信息。"""

    path: str
    label: str
    image: Optional[np.ndarray] = field(default=None, repr=False)
    height: int = 0
    width: int = 0
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass
class MatchResult:
    """单张目标图的匹配结果。"""

    image_path: str
    success: bool
    boxes: List[BoundingBox] = field(default_factory=list)
    error_message: str = ""


class TemplateMatcher:
    """基于 OpenCV 的多模板匹配器。

    Parameters
    ----------
    threshold : float
        匹配分数阈值。对于 TM_CCOEFF_NORMED 等方法，>= threshold 的候选框
        被保留；对于 TM_SQDIFF/TM_SQDIFF_NORMED，<= threshold 的候选框
        被保留（值越小匹配越好），置信度会被归一化为越大越好。
    max_candidates : int
        每张图片最多保留的候选框数（NMS 之后）。
    nms_iou_threshold : float
        NMS 的 IoU 阈值。
    method : int
        cv2.matchTemplate 使用的匹配算法，默认 TM_CCOEFF_NORMED。
    multi_scale : bool
        是否在多个缩放尺度上搜索。
    scale_range : tuple
        缩放搜索范围 (min_scale, max_scale)。
    scale_steps : int
        缩放搜索步数。
    """

    _SQDIFF_DEFAULT_THRESHOLD = 0.2

    def __init__(
        self,
        threshold: Optional[float] = None,
        max_candidates: int = 50,
        nms_iou_threshold: float = 0.3,
        method: int = cv2.TM_CCOEFF_NORMED,
        multi_scale: bool = False,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        scale_steps: int = 10,
    ):
        if threshold is None:
            threshold = (self._SQDIFF_DEFAULT_THRESHOLD
                         if method in _SQDIFF_METHODS else 0.8)
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.nms_iou_threshold = nms_iou_threshold
        self.method = method
        self.multi_scale = multi_scale
        self.scale_range = scale_range
        self.scale_steps = scale_steps

    @property
    def _is_sqdiff(self) -> bool:
        return self.method in _SQDIFF_METHODS

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_template(path: str, label: str) -> TemplateInfo:
        """加载模板图片并返回 TemplateInfo。

        Raises
        ------
        ValueError
            文件不存在或无法解码。
        """
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"模板文件不存在: {path}")

        img = imread_unicode(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取模板图片: {path}")

        h, w = img.shape[:2]
        return TemplateInfo(path=path, label=label, image=img, height=h, width=w)

    @staticmethod
    def create_template_from_image(
        image: np.ndarray,
        label: str,
        path: str = "",
        preprocess: Optional[PreprocessConfig] = None,
    ) -> TemplateInfo:
        """从内存中的 numpy 数组直接创建 TemplateInfo（用于裁剪后的模板）。"""
        h, w = image.shape[:2]
        return TemplateInfo(
            path=path,
            label=label,
            image=image,
            height=h,
            width=w,
            preprocess=preprocess or PreprocessConfig(),
        )

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_image(
        image: np.ndarray, config: PreprocessConfig
    ) -> np.ndarray:
        """按 PreprocessConfig 对图像施加预处理流水线。

        顺序: 灰度化 -> 高斯模糊 -> 二值化 -> Canny。
        """
        img = image.copy()

        if config.needs_grayscale and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if config.gaussian_blur_ksize > 0:
            k = config.gaussian_blur_ksize
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        if config.use_adaptive_threshold:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            block = config.adaptive_block_size
            if block % 2 == 0:
                block += 1
            block = max(3, block)
            img = cv2.adaptiveThreshold(
                img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if config.binary_inverse else cv2.THRESH_BINARY,
                block,
                config.adaptive_c,
            )
        elif config.binary_threshold >= 0:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_type = (
                cv2.THRESH_BINARY_INV if config.binary_inverse else cv2.THRESH_BINARY
            )
            _, img = cv2.threshold(img, config.binary_threshold, 255, thresh_type)

        if config.canny_enabled:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(img, config.canny_low, config.canny_high)

        return img

    @staticmethod
    def _extract_roi(
        image: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, int, int]:
        """从 image 中提取 ROI 子区域，返回 (子区域, offset_x, offset_y)。"""
        x, y, w, h = roi
        ih, iw = image.shape[:2]
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = min(w, iw - x)
        h = min(h, ih - y)
        if w <= 0 or h <= 0:
            logger.warning(
                "ROI 裁剪后区域为空 (roi=%s, image=%dx%d), 回退到全图",
                roi, iw, ih,
            )
            return image.copy(), 0, 0
        return image[y : y + h, x : x + w].copy(), x, y

    # ------------------------------------------------------------------
    # Single-image matching
    # ------------------------------------------------------------------

    def preprocess_templates(
        self, templates: List[TemplateInfo]
    ) -> Dict[int, np.ndarray]:
        """预处理所有模板图像并返回缓存字典（id(tpl) -> processed_image）。

        在批量匹配场景中调用一次，然后传入 match() 避免重复预处理。
        """
        cache: Dict[int, np.ndarray] = {}
        for tpl in templates:
            if tpl.image is None:
                continue
            cache[id(tpl)] = self.preprocess_image(tpl.image, tpl.preprocess)
        return cache

    def match(
        self,
        target_path: str,
        templates: List[TemplateInfo],
        _preprocessed_templates: Optional[Dict[int, np.ndarray]] = None,
    ) -> MatchResult:
        """对单张目标图执行多模板匹配。

        Parameters
        ----------
        _preprocessed_templates : dict, optional
            由 preprocess_templates() 返回的预处理缓存，避免重复计算。
        """
        target = imread_unicode(target_path, cv2.IMREAD_COLOR)
        if target is None:
            return MatchResult(
                image_path=target_path,
                success=False,
                error_message=f"无法读取目标图片: {target_path}",
            )

        all_boxes: List[BoundingBox] = []

        for tpl in templates:
            if tpl.image is None:
                continue
            try:
                search_region = target
                offset_x, offset_y = 0, 0
                if tpl.preprocess.target_roi:
                    search_region, offset_x, offset_y = self._extract_roi(
                        target, tpl.preprocess.target_roi
                    )

                processed_target = self.preprocess_image(
                    search_region, tpl.preprocess
                )
                if _preprocessed_templates and id(tpl) in _preprocessed_templates:
                    processed_tpl = _preprocessed_templates[id(tpl)]
                else:
                    processed_tpl = self.preprocess_image(tpl.image, tpl.preprocess)

                if self.multi_scale:
                    boxes = self._match_multi_scale(
                        processed_target, processed_tpl, tpl, offset_x, offset_y
                    )
                else:
                    boxes = self._match_single_scale(
                        processed_target, processed_tpl, tpl, offset_x, offset_y
                    )
                all_boxes.extend(boxes)
            except Exception as exc:
                logger.warning(
                    "模板 %s 在 %s 上匹配出错: %s",
                    tpl.label,
                    target_path,
                    exc,
                    exc_info=True,
                )

        all_boxes = self._nms(all_boxes)

        if self.max_candidates and len(all_boxes) > self.max_candidates:
            all_boxes.sort(key=lambda b: b.confidence, reverse=True)
            all_boxes = all_boxes[: self.max_candidates]

        return MatchResult(image_path=target_path, success=True, boxes=all_boxes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_single_scale(
        self,
        target: np.ndarray,
        tpl_image: np.ndarray,
        tpl: TemplateInfo,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[BoundingBox]:
        th, tw = tpl_image.shape[:2]
        if target.shape[0] < th or target.shape[1] < tw:
            logger.debug(
                "目标图 (%dx%d) 小于模板 '%s' (%dx%d), 跳过",
                target.shape[1], target.shape[0], tpl.label, tw, th,
            )
            return []

        result = cv2.matchTemplate(target, tpl_image, self.method)
        return self._extract_boxes(
            result, tpl.width, tpl.height, tpl.label, offset_x, offset_y
        )

    def _match_multi_scale(
        self,
        target: np.ndarray,
        tpl_image: np.ndarray,
        tpl: TemplateInfo,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[BoundingBox]:
        boxes: List[BoundingBox] = []
        lo, hi = self.scale_range
        scales = np.linspace(lo, hi, self.scale_steps)
        if not np.any(np.isclose(scales, 1.0)):
            scales = np.sort(np.append(scales, 1.0))
        for scale in scales:
            new_w = max(1, int(tpl_image.shape[1] * scale))
            new_h = max(1, int(tpl_image.shape[0] * scale))
            if new_h > target.shape[0] or new_w > target.shape[1]:
                continue
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(
                tpl_image, (new_w, new_h), interpolation=interp
            )
            result = cv2.matchTemplate(target, resized, self.method)
            orig_w = max(1, int(tpl.width * scale))
            orig_h = max(1, int(tpl.height * scale))
            boxes.extend(
                self._extract_boxes(
                    result, orig_w, orig_h, tpl.label, offset_x, offset_y
                )
            )
        return boxes

    def _extract_boxes(
        self,
        result: np.ndarray,
        tw: int,
        th: int,
        label: str,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[BoundingBox]:
        if self._is_sqdiff:
            locations = np.where(result <= self.threshold)
        else:
            locations = np.where(result >= self.threshold)
        boxes: List[BoundingBox] = []
        for y, x in zip(*locations):
            raw = float(result[y, x])
            if self._is_sqdiff:
                if self.method == cv2.TM_SQDIFF_NORMED:
                    confidence = 1.0 - raw
                else:
                    confidence = 1.0 / (1.0 + raw)
            else:
                confidence = raw
            boxes.append(
                BoundingBox(
                    label=label,
                    x_min=int(x) + offset_x,
                    y_min=int(y) + offset_y,
                    x_max=int(x + tw) + offset_x,
                    y_max=int(y + th) + offset_y,
                    confidence=confidence,
                )
            )
        return boxes

    def _nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """非极大值抑制（按标签分组，避免跨类别抑制）。"""
        if not boxes:
            return boxes

        label_groups: Dict[str, List[int]] = {}
        for i, b in enumerate(boxes):
            label_groups.setdefault(b.label, []).append(i)

        kept: List[BoundingBox] = []
        for group_indices in label_groups.values():
            group = [boxes[i] for i in group_indices]
            rects = [
                [float(b.x_min), float(b.y_min),
                 float(b.x_max - b.x_min), float(b.y_max - b.y_min)]
                for b in group
            ]
            scores = [float(b.confidence) for b in group]

            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=rects,
                scores=scores,
                score_threshold=0.0,
                nms_threshold=self.nms_iou_threshold,
            )

            if len(nms_indices) > 0:
                for idx in nms_indices.flatten().tolist():
                    kept.append(group[idx])

        return kept
