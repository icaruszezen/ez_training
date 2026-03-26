"""模板匹配引擎回归测试。

覆盖:
- NMS 按标签分组（不同标签不互相抑制，同标签高 IoU 合并）
- TM_CCOEFF_NORMED / TM_SQDIFF_NORMED 阈值方向正确性
- ROI 越界 / 零宽高 不崩溃且回退到全图
- 基础端到端匹配流程
"""

import cv2
import numpy as np
import pytest

from ez_training.prelabeling.models import BoundingBox
from ez_training.template_matching.matcher import (
    PreprocessConfig,
    TemplateMatcher,
    TemplateInfo,
)


def _make_solid(w: int, h: int, color=(128, 128, 128)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _stamp(canvas: np.ndarray, patch: np.ndarray, x: int, y: int):
    ph, pw = patch.shape[:2]
    canvas[y : y + ph, x : x + pw] = patch


def _save_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


# ======================================================================
# NMS
# ======================================================================


class TestNMSPerLabel:
    """NMS 按标签分组：不同标签的重叠框不应互相抑制。"""

    def test_different_labels_preserved(self):
        boxes = [
            BoundingBox(label="A", x_min=10, y_min=10, x_max=60, y_max=60, confidence=0.9),
            BoundingBox(label="B", x_min=15, y_min=15, x_max=65, y_max=65, confidence=0.85),
        ]
        matcher = TemplateMatcher(threshold=0.5, nms_iou_threshold=0.3)
        kept = matcher._nms(boxes)
        labels = {b.label for b in kept}
        assert labels == {"A", "B"}

    def test_same_label_suppressed(self):
        boxes = [
            BoundingBox(label="A", x_min=10, y_min=10, x_max=60, y_max=60, confidence=0.95),
            BoundingBox(label="A", x_min=12, y_min=12, x_max=62, y_max=62, confidence=0.80),
        ]
        matcher = TemplateMatcher(threshold=0.5, nms_iou_threshold=0.3)
        kept = matcher._nms(boxes)
        assert len(kept) == 1
        assert kept[0].confidence == 0.95

    def test_empty_input(self):
        matcher = TemplateMatcher()
        assert matcher._nms([]) == []

    def test_non_overlapping_same_label(self):
        boxes = [
            BoundingBox(label="A", x_min=0, y_min=0, x_max=30, y_max=30, confidence=0.9),
            BoundingBox(label="A", x_min=100, y_min=100, x_max=130, y_max=130, confidence=0.85),
        ]
        matcher = TemplateMatcher(threshold=0.5, nms_iou_threshold=0.3)
        kept = matcher._nms(boxes)
        assert len(kept) == 2


# ======================================================================
# Threshold direction
# ======================================================================


class TestThresholdDirection:

    def _make_pair(self):
        tpl = _make_solid(30, 30, (0, 200, 0))
        target = _make_solid(120, 120, (200, 200, 200))
        _stamp(target, tpl, 45, 45)
        return target, tpl

    def test_ccoeff_normed_finds_match(self, tmp_path):
        target, tpl_img = self._make_pair()
        target_path = tmp_path / "target.png"
        _save_png(target_path, target)

        tpl = TemplateMatcher.create_template_from_image(tpl_img, "green")
        matcher = TemplateMatcher(threshold=0.8, method=cv2.TM_CCOEFF_NORMED)
        result = matcher.match(str(target_path), [tpl])

        assert result.success
        assert len(result.boxes) >= 1
        assert all(b.confidence >= 0.8 for b in result.boxes)

    def test_sqdiff_normed_finds_match(self, tmp_path):
        target, tpl_img = self._make_pair()
        target_path = tmp_path / "target.png"
        _save_png(target_path, target)

        tpl = TemplateMatcher.create_template_from_image(tpl_img, "green")
        matcher = TemplateMatcher(threshold=0.1, method=cv2.TM_SQDIFF_NORMED)
        result = matcher.match(str(target_path), [tpl])

        assert result.success
        assert len(result.boxes) >= 1
        # confidence = 1 - raw_score; raw ~ 0 for perfect match → confidence ~ 1
        assert all(b.confidence > 0.5 for b in result.boxes)

    def test_sqdiff_normed_high_threshold_excludes(self, tmp_path):
        """threshold=0 应排除所有非完美匹配。"""
        target = _make_solid(120, 120, (200, 200, 200))
        target_path = tmp_path / "target.png"
        _save_png(target_path, target)

        tpl_img = _make_solid(30, 30, (0, 0, 255))
        tpl = TemplateMatcher.create_template_from_image(tpl_img, "blue")
        matcher = TemplateMatcher(threshold=0.0, method=cv2.TM_SQDIFF_NORMED)
        result = matcher.match(str(target_path), [tpl])

        assert result.success
        # threshold=0 means only raw_score <= 0 passes (perfect match only)
        # blue template on gray target → no perfect match
        assert len(result.boxes) == 0


# ======================================================================
# ROI edge cases
# ======================================================================


class TestROIEdgeCases:

    def test_roi_completely_outside(self):
        img = _make_solid(100, 100)
        region, ox, oy = TemplateMatcher._extract_roi(img, (200, 200, 50, 50))
        assert region.shape[0] > 0 and region.shape[1] > 0

    def test_roi_zero_size_fallback(self):
        img = _make_solid(100, 100)
        region, ox, oy = TemplateMatcher._extract_roi(img, (50, 50, 0, 0))
        assert region.shape[:2] == (100, 100)
        assert ox == 0 and oy == 0

    def test_roi_normal_crop(self):
        img = _make_solid(100, 100)
        region, ox, oy = TemplateMatcher._extract_roi(img, (10, 20, 30, 40))
        assert region.shape == (40, 30, 3)
        assert ox == 10 and oy == 20

    def test_roi_partial_overlap(self):
        img = _make_solid(100, 100)
        region, ox, oy = TemplateMatcher._extract_roi(img, (80, 80, 50, 50))
        assert region.shape[0] == 20
        assert region.shape[1] == 20


# ======================================================================
# End-to-end matching
# ======================================================================


class TestEndToEnd:

    def test_unreadable_target(self, tmp_path):
        tpl_img = _make_solid(20, 20, (255, 0, 0))
        tpl = TemplateMatcher.create_template_from_image(tpl_img, "red")
        matcher = TemplateMatcher()
        result = matcher.match(str(tmp_path / "nonexistent.png"), [tpl])

        assert not result.success
        assert "无法读取" in result.error_message

    def test_template_with_no_image_skipped(self, tmp_path):
        target = _make_solid(100, 100)
        target_path = tmp_path / "target.png"
        _save_png(target_path, target)

        tpl = TemplateInfo(path="", label="empty", image=None, height=0, width=0)
        matcher = TemplateMatcher()
        result = matcher.match(str(target_path), [tpl])

        assert result.success
        assert len(result.boxes) == 0

    def test_target_smaller_than_template(self, tmp_path):
        target = _make_solid(10, 10)
        target_path = tmp_path / "small.png"
        _save_png(target_path, target)

        tpl_img = _make_solid(50, 50, (0, 0, 255))
        tpl = TemplateMatcher.create_template_from_image(tpl_img, "big")
        matcher = TemplateMatcher()
        result = matcher.match(str(target_path), [tpl])

        assert result.success
        assert len(result.boxes) == 0

    def test_roi_based_matching(self, tmp_path):
        tpl_img = _make_solid(20, 20, (0, 0, 255))
        target = _make_solid(200, 200, (180, 180, 180))
        _stamp(target, tpl_img, 150, 150)
        target_path = tmp_path / "target.png"
        _save_png(target_path, target)

        config = PreprocessConfig(target_roi=(130, 130, 70, 70))
        tpl = TemplateMatcher.create_template_from_image(
            tpl_img, "blue", preprocess=config,
        )
        matcher = TemplateMatcher(threshold=0.8)
        result = matcher.match(str(target_path), [tpl])

        assert result.success
        assert len(result.boxes) >= 1
        for box in result.boxes:
            assert box.x_min >= 130
            assert box.y_min >= 130
