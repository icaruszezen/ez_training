"""数据模型单元测试"""

from dataclasses import asdict

from ez_training.prelabeling.models import (
    BoundingBox,
    DetectionMode,
    DetectionResult,
    PrelabelingStats,
    ReferenceImageInfo,
    VisionAPIConfig,
)


class TestVisionAPIConfig:
    """VisionAPIConfig 数据类测试"""

    def test_default_values(self):
        config = VisionAPIConfig()
        assert config.endpoint == ""
        assert config.api_key == ""
        assert config.model_name == "gpt-4-vision-preview"
        assert config.timeout == 60

    def test_custom_values(self):
        config = VisionAPIConfig(
            endpoint="https://api.example.com/v1",
            api_key="sk-test123",
            model_name="custom-model",
            timeout=120,
        )
        assert config.endpoint == "https://api.example.com/v1"
        assert config.api_key == "sk-test123"
        assert config.model_name == "custom-model"
        assert config.timeout == 120

    def test_serializable(self):
        config = VisionAPIConfig(endpoint="https://api.example.com", api_key="key")
        d = asdict(config)
        assert d == {
            "endpoint": "https://api.example.com",
            "api_key": "key",
            "model_name": "gpt-4-vision-preview",
            "timeout": 60,
        }


class TestBoundingBox:
    """BoundingBox 数据类测试"""

    def test_required_fields(self):
        box = BoundingBox(label="cat", x_min=10, y_min=20, x_max=100, y_max=200)
        assert box.label == "cat"
        assert box.x_min == 10
        assert box.y_min == 20
        assert box.x_max == 100
        assert box.y_max == 200
        assert box.confidence == 1.0

    def test_custom_confidence(self):
        box = BoundingBox(
            label="dog", x_min=0, y_min=0, x_max=50, y_max=50, confidence=0.85
        )
        assert box.confidence == 0.85

    def test_serializable(self):
        box = BoundingBox(label="car", x_min=1, y_min=2, x_max=3, y_max=4)
        d = asdict(box)
        assert d["label"] == "car"
        assert d["x_min"] == 1
        assert d["confidence"] == 1.0


class TestDetectionResult:
    """DetectionResult 数据类测试"""

    def test_success_result(self):
        boxes = [BoundingBox(label="a", x_min=0, y_min=0, x_max=10, y_max=10)]
        result = DetectionResult(success=True, boxes=boxes)
        assert result.success is True
        assert len(result.boxes) == 1
        assert result.error_message == ""
        assert result.raw_response == ""

    def test_failure_result(self):
        result = DetectionResult(
            success=False, error_message="timeout", raw_response='{"error": "timeout"}'
        )
        assert result.success is False
        assert result.boxes == []
        assert result.error_message == "timeout"

    def test_default_boxes_not_shared(self):
        """确保默认 boxes 列表不会在实例间共享"""
        r1 = DetectionResult(success=True)
        r2 = DetectionResult(success=True)
        r1.boxes.append(BoundingBox(label="x", x_min=0, y_min=0, x_max=1, y_max=1))
        assert len(r2.boxes) == 0


class TestPrelabelingStats:
    """PrelabelingStats 数据类测试"""

    def test_default_values(self):
        stats = PrelabelingStats()
        assert stats.total == 0
        assert stats.processed == 0
        assert stats.success == 0
        assert stats.failed == 0
        assert stats.skipped == 0

    def test_custom_values(self):
        stats = PrelabelingStats(
            total=100, processed=80, success=70, failed=10, skipped=20
        )
        assert stats.total == 100
        assert stats.processed == 80
        assert stats.success == 70
        assert stats.failed == 10
        assert stats.skipped == 20

    def test_consistency_invariant(self):
        """验证 Property 11: total = success + failed + skipped, processed = success + failed"""
        stats = PrelabelingStats(
            total=50, processed=30, success=25, failed=5, skipped=20
        )
        assert stats.total == stats.success + stats.failed + stats.skipped
        assert stats.processed == stats.success + stats.failed


class TestDetectionMode:
    """DetectionMode 枚举测试"""

    def test_text_only_value(self):
        assert DetectionMode.TEXT_ONLY.value == "text_only"

    def test_reference_image_value(self):
        assert DetectionMode.REFERENCE_IMAGE.value == "reference_image"

    def test_enum_members(self):
        members = list(DetectionMode)
        assert len(members) == 2
        assert DetectionMode.TEXT_ONLY in members
        assert DetectionMode.REFERENCE_IMAGE in members


class TestReferenceImageInfo:
    """ReferenceImageInfo 数据类测试"""

    def test_required_path(self):
        info = ReferenceImageInfo(path="/tmp/ref.jpg")
        assert info.path == "/tmp/ref.jpg"
        assert info.is_valid is True
        assert info.error_message == ""

    def test_invalid_image(self):
        info = ReferenceImageInfo(
            path="/tmp/bad.txt", is_valid=False, error_message="Unsupported format"
        )
        assert info.is_valid is False
        assert info.error_message == "Unsupported format"

    def test_serializable(self):
        info = ReferenceImageInfo(path="/tmp/ref.png")
        d = asdict(info)
        assert d == {"path": "/tmp/ref.png", "is_valid": True, "error_message": ""}
