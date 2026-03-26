import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# 旧版拼写错误的配置目录名；首次启动时自动迁移到 .ez_training
_LEGACY_CONFIG_DIR_NAME = ".ez_traing"

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif", ".avif"}

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _merge_missing_files(src_root: Path, dst_root: Path) -> None:
    """将 src 中存在的文件复制到 dst，若目标路径尚不存在。"""
    if not src_root.is_dir():
        return
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root)
        dest = dst_root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _maybe_migrate_legacy_config_dir(new: Path) -> None:
    """若存在旧目录 ~/.ez_traing，则整体改名或合并到新目录 ~/.ez_training。"""
    old = Path.home() / _LEGACY_CONFIG_DIR_NAME
    if not old.exists():
        return
    try:
        if not new.exists():
            shutil.move(str(old), str(new))
            logger.info("已将配置目录从 %s 迁移到 %s", old, new)
            return
        _merge_missing_files(old, new)
        logger.info("已从 %s 合并缺失的配置文件到 %s（旧目录仍保留，确认后可手动删除）", old, new)
    except OSError as exc:
        logger.warning("迁移旧配置目录失败 (%s -> %s): %s", old, new, exc)


def get_config_dir() -> Path:
    config_dir = Path.home() / ".ez_training"
    _maybe_migrate_legacy_config_dir(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


_SETTINGS_FILE = "settings.json"

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "github_mirror_enabled": False,
    "github_mirror_url": "https://ghp.ci/",
    "sample_dataset_dir": "",
}


def load_settings() -> Dict[str, Any]:
    path = get_config_dir() / _SETTINGS_FILE
    settings = dict(_DEFAULT_SETTINGS)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s, falling back to defaults: %s", path, exc)
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    path = get_config_dir() / _SETTINGS_FILE
    merged = dict(_DEFAULT_SETTINGS)
    merged.update(settings)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_github_mirror_prefix() -> str:
    """Return the mirror URL prefix if enabled, otherwise empty string."""
    settings = load_settings()
    if settings.get("github_mirror_enabled"):
        url = settings.get("github_mirror_url", "")
        if url and not url.endswith("/"):
            url += "/"
        return url
    return ""


def detect_devices() -> List[Tuple[str, str]]:
    """Return ``[(device_id, display_name), ...]`` for available GPUs then CPU."""
    devices: List[Tuple[str, str]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                devices.append((str(i), f"GPU {i}: {name} ({memory_gb:.1f}GB)"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(("mps", "MPS (Apple Silicon)"))
    except Exception:
        pass
    devices.append(("cpu", "CPU"))
    return devices


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def open_path(path: str) -> None:
    """Cross-platform open file/directory in the native file manager."""
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(path))
