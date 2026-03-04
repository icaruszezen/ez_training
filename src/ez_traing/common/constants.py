import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def get_config_dir() -> Path:
    config_dir = Path.home() / ".ez_traing"
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
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    path = get_config_dir() / _SETTINGS_FILE
    merged = dict(_DEFAULT_SETTINGS)
    merged.update(settings)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


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
    """Return ``[(device_id, display_name), ...]`` for CPU and available GPUs."""
    devices: List[Tuple[str, str]] = [("cpu", "CPU")]
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                devices.insert(0, (str(i), f"GPU {i}: {name} ({memory_gb:.1f}GB)"))
    except Exception:
        pass
    return devices


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def open_path(path: str) -> None:
    """Cross-platform open file/directory in the native file manager."""
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(path))
