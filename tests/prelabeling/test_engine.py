"""PrelabelingWorker 单元测试"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import pytest
from PIL import Image

from ez_traing.prelabeling.engine import PrelabelingWorker, validate_prelabeling_input
from ez_traing.prelabeling.models import (
    BoundingBox,
    DetectionMode,
    DetectionResult,
    PrelabelingStats,
)


@pytest.fixture
def mock_vision_service():
    """创建 mock 的 VisionModelService"""
    service = MagicMock()
    service.detect_objects.return_value = DetectionResult(
        success=True,
        boxes=[
            BoundingBox(label="cat", x_min=10, y_min=20, x_max=100, y_max=200),
        ],
    )
    return service


@pytest.fixture
def sample_images(tmp_path):
    """创建临时测试图片列表"""
    paths = []
    for i in range(3):
        img = Image.new("RGB", (640, 480), color="blue")
        img_path = tmp_path / f"image_{i}.jpg"
        img.save(str(img_path))
        paths.append(str(img_path))
    return paths


@pytest.fixture
def worker(sample_images, mock_vision_service):
    """创建 PrelabelingWorker 实例"""
    return PrelabelingWorker(
        image_paths=sample_images,
        prompt="detect cats",
        vision_service=mock_vision_service,
    )


class TestPrelabelingWorkerInit:
    """构造函数测试"""

    def test_stores_image_paths(self, worker, sample_images):
        assert worker._image_paths == sample_images

    def test_stores_prompt(self, worker):
        assert worker._prompt == "detect cats"

    def test_default_skip_annotated(self, worker):
        assert worker._skip_annotated is True

    def test_default_overwrite(self, worker):
        assert worker._overwrite is False

    def test_not_cancelled_initially(self, worker):
        assert worker._cancelled is False

    def test_custom_skip_annotated(self, sample_images, mock_vision_service):
        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service, skip_annotated=False
        )
        assert w._skip_annotated is False

    def test_custom_overwrite(self, sample_images, mock_vision_service):
        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service, overwrite=True
        )
        assert w._overwrite is True

    def test_default_detection_mode_is_text_only(self, worker):
        assert worker._detection_mode == DetectionMode.TEXT_ONLY

    def test_default_reference_images_is_empty_list(self, worker):
        assert worker._reference_images == []

    def test_custom_detection_mode_reference_image(self, sample_images, mock_vision_service):
        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service,
            reference_images=["/path/to/ref.jpg"],
            detection_mode="reference_image",
        )
        assert w._detection_mode == DetectionMode.REFERENCE_IMAGE

    def test_stores_reference_images(self, sample_images, mock_vision_service):
        refs = ["/path/ref1.jpg", "/path/ref2.png"]
        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service,
            reference_images=refs,
            detection_mode="reference_image",
        )
        assert w._reference_images == refs

    def test_reference_image_mode_without_images_raises(self, sample_images, mock_vision_service):
        with pytest.raises(ValueError, match="参考图片模式下必须提供至少一张参考图片"):
            PrelabelingWorker(
                sample_images, "prompt", mock_vision_service,
                detection_mode="reference_image",
            )

    def test_reference_image_mode_with_empty_list_raises(self, sample_images, mock_vision_service):
        with pytest.raises(ValueError, match="参考图片模式下必须提供至少一张参考图片"):
            PrelabelingWorker(
                sample_images, "prompt", mock_vision_service,
                reference_images=[],
                detection_mode="reference_image",
            )

    def test_invalid_detection_mode_raises(self, sample_images, mock_vision_service):
        with pytest.raises(ValueError):
            PrelabelingWorker(
                sample_images, "prompt", mock_vision_service,
                detection_mode="invalid_mode",
            )

    def test_text_only_mode_with_reference_images_stores_them(self, sample_images, mock_vision_service):
        refs = ["/path/ref.jpg"]
        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service,
            reference_images=refs,
            detection_mode="text_only",
        )
        assert w._reference_images == refs
        assert w._detection_mode == DetectionMode.TEXT_ONLY


class TestCancel:
    """cancel 方法测试"""

    def test_sets_cancelled_flag(self, worker):
        worker.cancel()
        assert worker._cancelled is True

    def test_cancel_idempotent(self, worker):
        worker.cancel()
        worker.cancel()
        assert worker._cancelled is True


class TestHasAnnotation:
    """_has_annotation 方法测试"""

    def test_no_annotation(self, worker, sample_images):
        assert worker._has_annotation(sample_images[0]) is False

    def test_has_annotation(self, worker, sample_images):
        xml_path = Path(sample_images[0]).with_suffix(".xml")
        xml_path.write_text("<annotation/>")
        assert worker._has_annotation(sample_images[0]) is True

    def test_different_extension(self, worker, tmp_path):
        img_path = tmp_path / "photo.png"
        img_path.write_bytes(b"fake")
        xml_path = tmp_path / "photo.xml"
        xml_path.write_text("<annotation/>")
        assert worker._has_annotation(str(img_path)) is True


class TestRun:
    """run 方法测试（直接调用，不启动线程）"""

    def test_processes_all_images(self, worker, sample_images, mock_vision_service):
        """处理所有图片"""
        finished_stats = []
        worker.finished.connect(lambda s: finished_stats.append(s))

        worker.run()

        assert mock_vision_service.detect_objects.call_count == 3
        assert len(finished_stats) == 1
        stats = finished_stats[0]
        assert stats.total == 3
        assert stats.success == 3
        assert stats.failed == 0
        assert stats.skipped == 0

    def test_emits_progress_for_each_image(self, worker, sample_images):
        """每张图片都发射 progress 信号"""
        progress_calls = []
        worker.progress.connect(lambda c, t, m: progress_calls.append((c, t, m)))

        worker.run()

        assert len(progress_calls) == 3
        for i, (current, total, _msg) in enumerate(progress_calls):
            assert current == i + 1
            assert total == 3

    def test_emits_image_completed_for_each(self, worker, sample_images):
        """每张图片都发射 image_completed 信号"""
        completed_calls = []
        worker.image_completed.connect(
            lambda p, s, m: completed_calls.append((p, s, m))
        )

        worker.run()

        assert len(completed_calls) == 3
        for path, success, _msg in completed_calls:
            assert success is True

    def test_emits_finished_with_stats(self, worker):
        """完成时发射 finished 信号"""
        finished_stats = []
        worker.finished.connect(lambda s: finished_stats.append(s))

        worker.run()

        assert len(finished_stats) == 1
        assert isinstance(finished_stats[0], PrelabelingStats)

    def test_skips_annotated_images(
        self, sample_images, mock_vision_service
    ):
        """skip_annotated=True 时跳过已标注图片"""
        # 为第一张图片创建标注文件
        xml_path = Path(sample_images[0]).with_suffix(".xml")
        xml_path.write_text("<annotation/>")

        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service, skip_annotated=True
        )
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.skipped == 1
        assert stats.success == 2
        assert mock_vision_service.detect_objects.call_count == 2

    def test_does_not_skip_when_disabled(
        self, sample_images, mock_vision_service
    ):
        """skip_annotated=False 时不跳过已标注图片"""
        xml_path = Path(sample_images[0]).with_suffix(".xml")
        xml_path.write_text("<annotation/>")

        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service, skip_annotated=False
        )
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.skipped == 0
        assert mock_vision_service.detect_objects.call_count == 3

    def test_overwrite_processes_annotated(
        self, sample_images, mock_vision_service
    ):
        """overwrite=True 时处理已标注图片"""
        xml_path = Path(sample_images[0]).with_suffix(".xml")
        xml_path.write_text("<annotation/>")

        w = PrelabelingWorker(
            sample_images,
            "prompt",
            mock_vision_service,
            skip_annotated=True,
            overwrite=True,
        )
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.skipped == 0
        assert mock_vision_service.detect_objects.call_count == 3

    def test_cancel_stops_processing(
        self, sample_images, mock_vision_service
    ):
        """取消后停止处理后续图片"""
        call_count = 0

        def detect_and_cancel(path, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一张图片处理后取消
                w.cancel()
            return DetectionResult(
                success=True,
                boxes=[BoundingBox("cat", 0, 0, 10, 10)],
            )

        mock_vision_service.detect_objects.side_effect = detect_and_cancel

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        # 第一张处理完后设置取消，第二张开始时检测到取消标志
        assert call_count <= 2
        stats = finished_stats[0]
        assert stats.processed < stats.total

    def test_detection_failure_increments_failed(
        self, sample_images, mock_vision_service
    ):
        """检测失败时 failed 计数增加"""
        mock_vision_service.detect_objects.return_value = DetectionResult(
            success=False, error_message="API error"
        )

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.failed == 3
        assert stats.success == 0

    def test_failure_emits_image_completed_false(
        self, sample_images, mock_vision_service
    ):
        """检测失败时 image_completed 信号 success=False"""
        mock_vision_service.detect_objects.return_value = DetectionResult(
            success=False, error_message="timeout"
        )

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        completed = []
        w.image_completed.connect(lambda p, s, m: completed.append((p, s, m)))

        w.run()

        for _path, success, msg in completed:
            assert success is False
            assert "timeout" in msg

    def test_empty_boxes_counts_as_success(
        self, sample_images, mock_vision_service
    ):
        """检测成功但无目标框时计为 success"""
        mock_vision_service.detect_objects.return_value = DetectionResult(
            success=True, boxes=[]
        )

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.success == 3
        assert stats.failed == 0

    def test_mixed_results(self, sample_images, mock_vision_service):
        """混合成功和失败的结果"""
        results = [
            DetectionResult(
                success=True,
                boxes=[BoundingBox("cat", 0, 0, 10, 10)],
            ),
            DetectionResult(success=False, error_message="error"),
            DetectionResult(
                success=True,
                boxes=[BoundingBox("dog", 5, 5, 15, 15)],
            ),
        ]
        mock_vision_service.detect_objects.side_effect = results

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.success == 2
        assert stats.failed == 1
        assert stats.processed == 3

    def test_saves_voc_annotation(self, sample_images, mock_vision_service):
        """成功检测后保存 VOC 标注文件"""
        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        w.run()

        for img_path in sample_images:
            xml_path = Path(img_path).with_suffix(".xml")
            assert xml_path.exists(), f"标注文件不存在: {xml_path}"
            tree = ElementTree.parse(str(xml_path))
            root = tree.getroot()
            objects = root.findall("object")
            assert len(objects) == 1
            assert objects[0].find("name").text == "cat"

    def test_save_failure_increments_failed(
        self, sample_images, mock_vision_service
    ):
        """保存标注失败时 failed 计数增加"""
        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)

        # Mock _save_voc_annotation to raise
        original_save = w._save_voc_annotation
        call_idx = [0]

        def failing_save(path, boxes, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 2:
                raise OSError("Permission denied")
            return original_save(path, boxes, **kwargs)

        w._save_voc_annotation = failing_save

        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))
        w.run()

        stats = finished_stats[0]
        assert stats.failed == 1
        assert stats.success == 2

    def test_continues_after_failure(self, sample_images, mock_vision_service):
        """单张图片失败后继续处理后续图片（需求 8.5）"""
        results = [
            DetectionResult(success=False, error_message="fail1"),
            DetectionResult(
                success=True,
                boxes=[BoundingBox("cat", 0, 0, 10, 10)],
            ),
            DetectionResult(success=False, error_message="fail3"),
        ]
        mock_vision_service.detect_objects.side_effect = results

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        assert mock_vision_service.detect_objects.call_count == 3
        stats = finished_stats[0]
        assert stats.success == 1
        assert stats.failed == 2

    def test_stats_consistency(self, sample_images, mock_vision_service):
        """统计数据一致性: total = success + failed + skipped"""
        xml_path = Path(sample_images[0]).with_suffix(".xml")
        xml_path.write_text("<annotation/>")

        results = [
            # image_1 will be processed (image_0 is skipped)
            DetectionResult(
                success=True,
                boxes=[BoundingBox("cat", 0, 0, 10, 10)],
            ),
            DetectionResult(success=False, error_message="error"),
        ]
        mock_vision_service.detect_objects.side_effect = results

        w = PrelabelingWorker(sample_images, "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.total == stats.success + stats.failed + stats.skipped
        assert stats.processed == stats.success + stats.failed

    def test_empty_image_list(self, mock_vision_service):
        """空图片列表"""
        w = PrelabelingWorker([], "prompt", mock_vision_service)
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.total == 0
        assert stats.processed == 0
        assert stats.success == 0
        assert stats.failed == 0
        assert stats.skipped == 0


class TestReferenceImageMode:
    """参考图片模式下 _process_one 调用分支测试 (需求 6.2, 6.4)"""

    @pytest.fixture
    def ref_images(self):
        return ["/path/ref1.jpg", "/path/ref2.png"]

    @pytest.fixture
    def ref_worker(self, sample_images, mock_vision_service, ref_images):
        """创建参考图片模式的 PrelabelingWorker"""
        mock_vision_service.detect_objects_with_reference.return_value = DetectionResult(
            success=True,
            boxes=[BoundingBox(label="target", x_min=5, y_min=10, x_max=50, y_max=60)],
        )
        return PrelabelingWorker(
            image_paths=sample_images,
            prompt="find similar objects",
            vision_service=mock_vision_service,
            reference_images=ref_images,
            detection_mode="reference_image",
        )

    def test_reference_mode_calls_detect_with_reference(
        self, ref_worker, sample_images, mock_vision_service, ref_images
    ):
        """参考图片模式应调用 detect_objects_with_reference"""
        ref_worker.run()

        assert mock_vision_service.detect_objects_with_reference.call_count == 3
        mock_vision_service.detect_objects.assert_not_called()

    def test_reference_mode_passes_reference_images(
        self, ref_worker, sample_images, mock_vision_service, ref_images
    ):
        """每次调用都传递相同的参考图片列表 (需求 6.4)"""
        ref_worker.run()

        for call in mock_vision_service.detect_objects_with_reference.call_args_list:
            assert call.args[0] == ref_images

    def test_reference_mode_passes_target_path(
        self, ref_worker, sample_images, mock_vision_service
    ):
        """每次调用传递正确的待检测图片路径"""
        ref_worker.run()

        called_targets = [
            call.args[1]
            for call in mock_vision_service.detect_objects_with_reference.call_args_list
        ]
        assert called_targets == sample_images

    def test_reference_mode_passes_prompt(
        self, ref_worker, mock_vision_service
    ):
        """每次调用传递用户描述作为 user_description"""
        ref_worker.run()

        for call in mock_vision_service.detect_objects_with_reference.call_args_list:
            assert call.args[2] == "find similar objects"

    def test_text_only_mode_calls_detect_objects(
        self, worker, sample_images, mock_vision_service
    ):
        """文本模式应调用 detect_objects 而非 detect_objects_with_reference"""
        worker.run()

        assert mock_vision_service.detect_objects.call_count == 3
        mock_vision_service.detect_objects_with_reference.assert_not_called()

    def test_reference_mode_detection_failure(
        self, sample_images, mock_vision_service, ref_images
    ):
        """参考图片模式下检测失败正确处理"""
        mock_vision_service.detect_objects_with_reference.return_value = DetectionResult(
            success=False, error_message="参考图片编码失败"
        )
        w = PrelabelingWorker(
            sample_images, "prompt", mock_vision_service,
            reference_images=ref_images,
            detection_mode="reference_image",
        )
        finished_stats = []
        w.finished.connect(lambda s: finished_stats.append(s))

        w.run()

        stats = finished_stats[0]
        assert stats.failed == 3
        assert stats.success == 0

    def test_reference_images_same_across_all_calls(
        self, ref_worker, sample_images, mock_vision_service, ref_images
    ):
        """整个批量处理过程中复用相同的参考图片列表 (需求 6.4)"""
        ref_worker.run()

        ref_lists = [
            call.args[0]
            for call in mock_vision_service.detect_objects_with_reference.call_args_list
        ]
        # All calls should receive the exact same list object
        assert all(r is ref_lists[0] for r in ref_lists)


class TestValidatePrelabelingInput:
    """validate_prelabeling_input 函数测试"""

    def _make_config_manager(self, configured: bool):
        """创建 mock 的 APIConfigManager"""
        mgr = MagicMock()
        mgr.is_configured.return_value = configured
        return mgr

    # --- 提示词非空验证 (需求 2.4) ---

    def test_empty_string_raises(self):
        mgr = self._make_config_manager(True)
        with pytest.raises(ValueError, match="提示词不能为空"):
            validate_prelabeling_input("", mgr)

    def test_whitespace_only_raises(self):
        mgr = self._make_config_manager(True)
        with pytest.raises(ValueError, match="提示词不能为空"):
            validate_prelabeling_input("   \t\n  ", mgr)

    def test_none_prompt_raises(self):
        mgr = self._make_config_manager(True)
        with pytest.raises(ValueError, match="提示词不能为空"):
            validate_prelabeling_input(None, mgr)

    # --- 配置完整性验证 (需求 8.1) ---

    def test_unconfigured_raises(self):
        mgr = self._make_config_manager(False)
        with pytest.raises(ValueError, match="API 配置不完整"):
            validate_prelabeling_input("detect cats", mgr)

    # --- 验证通过 ---

    def test_valid_input_returns_true(self):
        mgr = self._make_config_manager(True)
        assert validate_prelabeling_input("detect cats", mgr) is True

    def test_prompt_with_leading_trailing_spaces_passes(self):
        mgr = self._make_config_manager(True)
        assert validate_prelabeling_input("  detect cats  ", mgr) is True

    # --- 验证顺序：提示词先于配置 ---

    def test_empty_prompt_checked_before_config(self):
        """空提示词应先于配置检查被捕获"""
        mgr = self._make_config_manager(False)
        with pytest.raises(ValueError, match="提示词不能为空"):
            validate_prelabeling_input("", mgr)

    # --- 参考图片模式验证 (需求 6.3) ---

    def test_reference_mode_no_images_raises(self):
        """参考图片模式下未提供参考图片应抛出错误"""
        mgr = self._make_config_manager(True)
        with pytest.raises(ValueError, match="参考图片模式下必须提供至少一张参考图片"):
            validate_prelabeling_input(
                "", mgr, detection_mode="reference_image", reference_images=None
            )

    def test_reference_mode_empty_list_raises(self):
        """参考图片模式下提供空列表应抛出错误"""
        mgr = self._make_config_manager(True)
        with pytest.raises(ValueError, match="参考图片模式下必须提供至少一张参考图片"):
            validate_prelabeling_input(
                "", mgr, detection_mode="reference_image", reference_images=[]
            )

    def test_reference_mode_with_images_passes(self):
        """参考图片模式下提供参考图片应通过验证"""
        mgr = self._make_config_manager(True)
        assert validate_prelabeling_input(
            "", mgr, detection_mode="reference_image",
            reference_images=["/path/to/ref.jpg"],
        ) is True

    def test_reference_mode_empty_prompt_passes(self):
        """参考图片模式下空提示词应通过（参考图片提供上下文）"""
        mgr = self._make_config_manager(True)
        assert validate_prelabeling_input(
            "", mgr, detection_mode="reference_image",
            reference_images=["/path/to/ref.jpg"],
        ) is True

    def test_reference_mode_with_prompt_and_images_passes(self):
        """参考图片模式下同时提供提示词和参考图片应通过"""
        mgr = self._make_config_manager(True)
        assert validate_prelabeling_input(
            "find similar objects", mgr, detection_mode="reference_image",
            reference_images=["/path/to/ref1.jpg", "/path/to/ref2.jpg"],
        ) is True

    def test_reference_mode_unconfigured_raises(self):
        """参考图片模式下 API 未配置应抛出错误"""
        mgr = self._make_config_manager(False)
        with pytest.raises(ValueError, match="API 配置不完整"):
            validate_prelabeling_input(
                "", mgr, detection_mode="reference_image",
                reference_images=["/path/to/ref.jpg"],
            )

    def test_reference_mode_no_images_checked_before_config(self):
        """参考图片模式下缺少参考图片应先于配置检查被捕获"""
        mgr = self._make_config_manager(False)
        with pytest.raises(ValueError, match="参考图片模式下必须提供至少一张参考图片"):
            validate_prelabeling_input(
                "", mgr, detection_mode="reference_image", reference_images=None
            )

    def test_text_only_mode_ignores_reference_images(self):
        """文本模式下即使提供参考图片也按文本模式验证"""
        mgr = self._make_config_manager(True)
        assert validate_prelabeling_input(
            "detect cats", mgr, detection_mode="text_only",
            reference_images=["/path/to/ref.jpg"],
        ) is True
