"""APIConfigManager 单元测试"""

import json

import pytest

from ez_training.prelabeling.config import APIConfigManager
from ez_training.prelabeling.models import VisionAPIConfig


@pytest.fixture
def config_dir(tmp_path):
    """提供临时配置目录"""
    return tmp_path / ".ez_training"


@pytest.fixture
def manager(config_dir):
    """提供 APIConfigManager 实例"""
    return APIConfigManager(config_dir=config_dir)


class TestAPIConfigManagerInit:
    """初始化测试"""

    def test_default_config(self, manager):
        config = manager.get_config()
        assert config.endpoint == ""
        assert config.api_key == ""
        assert config.model_name == "gpt-4-vision-preview"
        assert config.timeout == 60

    def test_creates_config_dir(self, config_dir):
        APIConfigManager(config_dir=config_dir)
        assert config_dir.exists()

    def test_loads_existing_config(self, config_dir):
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "vision_api_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "endpoint": "https://api.example.com/v1",
                    "api_key": "sk-test123456789",
                    "model_name": "custom-model",
                    "timeout": 120,
                }
            )
        )
        mgr = APIConfigManager(config_dir=config_dir)
        config = mgr.get_config()
        assert config.endpoint == "https://api.example.com/v1"
        assert config.api_key == "sk-test123456789"
        assert config.model_name == "custom-model"
        assert config.timeout == 120


class TestLoad:
    """加载配置测试"""

    def test_load_missing_file(self, manager):
        """配置文件不存在时使用默认值"""
        config = manager.get_config()
        assert config == VisionAPIConfig()

    def test_load_invalid_json(self, config_dir):
        """JSON 格式错误时使用默认值"""
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "vision_api_config.json").write_text("not json")
        mgr = APIConfigManager(config_dir=config_dir)
        assert mgr.get_config() == VisionAPIConfig()

    def test_load_partial_config(self, config_dir):
        """部分字段缺失时使用默认值填充"""
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "vision_api_config.json").write_text(
            json.dumps({"endpoint": "https://api.example.com"})
        )
        mgr = APIConfigManager(config_dir=config_dir)
        config = mgr.get_config()
        assert config.endpoint == "https://api.example.com"
        assert config.api_key == ""
        assert config.model_name == "gpt-4-vision-preview"
        assert config.timeout == 60


class TestSave:
    """保存配置测试"""

    def test_save_creates_file(self, manager, config_dir):
        manager.save()
        config_file = config_dir / "vision_api_config.json"
        assert config_file.exists()

    def test_save_content(self, manager, config_dir):
        manager.update_config(
            endpoint="https://api.example.com", api_key="sk-abc"
        )
        config_file = config_dir / "vision_api_config.json"
        data = json.loads(config_file.read_text())
        assert data["endpoint"] == "https://api.example.com"
        assert data["api_key"] == "sk-abc"


class TestUpdateConfig:
    """更新配置测试"""

    def test_update_single_field(self, manager):
        manager.update_config(endpoint="https://new.api.com")
        assert manager.get_config().endpoint == "https://new.api.com"
        assert manager.get_config().api_key == ""  # 未更新的字段保持不变

    def test_update_multiple_fields(self, manager):
        manager.update_config(endpoint="https://api.com", api_key="key123")
        config = manager.get_config()
        assert config.endpoint == "https://api.com"
        assert config.api_key == "key123"

    def test_update_saves_immediately(self, manager, config_dir):
        manager.update_config(endpoint="https://api.com")
        config_file = config_dir / "vision_api_config.json"
        data = json.loads(config_file.read_text())
        assert data["endpoint"] == "https://api.com"

    def test_update_none_does_not_change(self, manager):
        manager.update_config(endpoint="https://api.com", api_key="key")
        manager.update_config(endpoint=None, api_key=None)
        config = manager.get_config()
        assert config.endpoint == "https://api.com"
        assert config.api_key == "key"


class TestIsConfigured:
    """配置完整性验证测试"""

    def test_not_configured_by_default(self, manager):
        assert manager.is_configured() is False

    def test_not_configured_endpoint_only(self, manager):
        manager.update_config(endpoint="https://api.com")
        assert manager.is_configured() is False

    def test_not_configured_api_key_only(self, manager):
        manager.update_config(api_key="sk-key")
        assert manager.is_configured() is False

    def test_configured_with_both(self, manager):
        manager.update_config(endpoint="https://api.com", api_key="sk-key")
        assert manager.is_configured() is True


class TestGetMaskedApiKey:
    """API Key 脱敏测试"""

    def test_empty_key(self, manager):
        assert manager.get_masked_api_key() == ""

    def test_short_key_4_chars(self, manager):
        manager.update_config(api_key="abcd")
        assert manager.get_masked_api_key() == "****"

    def test_short_key_8_chars(self, manager):
        manager.update_config(api_key="abcdefgh")
        assert manager.get_masked_api_key() == "********"

    def test_key_9_chars(self, manager):
        manager.update_config(api_key="abcdefghi")
        assert manager.get_masked_api_key() == "abcd*fghi"

    def test_long_key(self, manager):
        manager.update_config(api_key="sk-1234567890abcdef")
        masked = manager.get_masked_api_key()
        assert len(masked) == len("sk-1234567890abcdef")
        assert masked[:4] == "sk-1"
        assert masked[-4:] == "cdef"
        assert all(c == "*" for c in masked[4:-4])

    def test_exactly_boundary(self, manager):
        """长度刚好为 8 时全部脱敏"""
        manager.update_config(api_key="12345678")
        assert manager.get_masked_api_key() == "********"


class TestRoundTrip:
    """配置保存-加载往返测试"""

    def test_roundtrip(self, config_dir):
        mgr1 = APIConfigManager(config_dir=config_dir)
        mgr1.update_config(
            endpoint="https://api.example.com/v1",
            api_key="sk-test-key-12345",
            model_name="gpt-4o",
            timeout=90,
        )
        mgr2 = APIConfigManager(config_dir=config_dir)
        config = mgr2.get_config()
        assert config.endpoint == "https://api.example.com/v1"
        assert config.api_key == "sk-test-key-12345"
        assert config.model_name == "gpt-4o"
        assert config.timeout == 90
