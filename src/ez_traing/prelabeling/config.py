"""API 配置管理器，负责存储和读取视觉模型 API 配置"""

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ez_traing.prelabeling.models import VisionAPIConfig

logger = logging.getLogger(__name__)


class APIConfigManager:
    """API 配置管理器

    负责 API 配置的持久化存储和读取。
    配置保存在 ~/.ez_traing/vision_api_config.json。
    """

    CONFIG_FILE = "vision_api_config.json"

    def __init__(self, config_dir: Optional[Path] = None):
        self._config: VisionAPIConfig = VisionAPIConfig()
        self._config_dir = config_dir or (Path.home() / ".ez_traing")
        self._config_path: Path = self._get_config_path()
        self.load()

    def _get_config_path(self) -> Path:
        """获取配置文件路径"""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        return self._config_dir / self.CONFIG_FILE

    def load(self) -> None:
        """从文件加载配置"""
        if not self._config_path.exists():
            logger.info("配置文件不存在，使用默认配置: %s", self._config_path)
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._config = VisionAPIConfig(
                endpoint=data.get("endpoint", ""),
                api_key=data.get("api_key", ""),
                model_name=data.get("model_name", "gpt-4-vision-preview"),
                timeout=data.get("timeout", 60),
            )
            logger.info("配置加载成功: %s", self._config_path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("配置加载失败，使用默认配置: %s", e)
            bak = self._config_path.with_suffix(".json.bak")
            try:
                self._config_path.rename(bak)
                logger.info("已将损坏的配置备份到 %s", bak)
            except OSError:
                pass
            self._config = VisionAPIConfig()

    def save(self) -> None:
        """保存配置到文件"""
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self._config), f, indent=2, ensure_ascii=False)
            if sys.platform != "win32":
                try:
                    os.chmod(self._config_path, 0o600)
                except OSError:
                    pass
            logger.info("配置保存成功: %s", self._config_path)
        except OSError as e:
            logger.error("配置保存失败: %s", e)
            raise

    def get_config(self) -> VisionAPIConfig:
        """获取当前配置"""
        return self._config

    def update_config(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """更新配置并立即保存

        仅更新传入的非 None 参数。
        """
        if endpoint is not None:
            self._config.endpoint = endpoint
        if api_key is not None:
            self._config.api_key = api_key
        if model_name is not None:
            self._config.model_name = model_name
        if timeout is not None:
            self._config.timeout = timeout
        self.save()

    def is_configured(self) -> bool:
        """检查配置是否完整（endpoint 和 api_key 都非空）"""
        return bool(self._config.endpoint and self._config.api_key)

    def get_masked_api_key(self) -> str:
        """获取脱敏的 API Key

        规则：
        - 长度 <= 8：全部替换为 *
        - 长度 > 8：保留前 4 位和后 4 位，中间替换为 *
        """
        key = self._config.api_key
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 8) + key[-4:]
