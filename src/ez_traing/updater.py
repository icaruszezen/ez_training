"""应用自动更新模块。

通过 GitHub Releases API 检查新版本，下载 zip 并通过 bat 脚本完成替换重启。
"""

import os
import sys
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional

import requests
from PyQt5.QtCore import QThread, pyqtSignal

from ez_traing import __version__
from ez_traing.common.constants import get_github_mirror_prefix

GITHUB_REPO = "icaruszezen/ez_traing"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
REQUEST_TIMEOUT = 15


def _mirror_url(url: str) -> str:
    """Prepend the configured mirror prefix to a URL when enabled."""
    prefix = get_github_mirror_prefix()
    if prefix:
        return prefix + url
    return url


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _current_exe_dir() -> str:
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _compare_versions(remote: str, local: str) -> bool:
    """Return True if *remote* is strictly newer than *local*.

    Supports simple numeric versions like ``0.2.1``.
    """
    def _parse(v: str):
        return tuple(int(x) for x in v.lstrip("vV").split("."))
    try:
        return _parse(remote) > _parse(local)
    except (ValueError, TypeError):
        return False


@dataclass
class ReleaseInfo:
    tag: str
    name: str
    body: str
    download_url: str
    size: int


class CheckUpdateWorker(QThread):
    """后台线程：检查 GitHub 是否有新版本。"""

    finished = pyqtSignal(object)  # ReleaseInfo | None
    error = pyqtSignal(str)

    def run(self):
        try:
            api_url = _mirror_url(RELEASES_API)
            try:
                resp = requests.get(
                    api_url,
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
            except requests.RequestException:
                if api_url != RELEASES_API:
                    resp = requests.get(
                        RELEASES_API,
                        headers={"Accept": "application/vnd.github.v3+json"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                else:
                    raise
            data = resp.json()

            tag = data.get("tag_name", "")
            if not _compare_versions(tag, __version__):
                self.finished.emit(None)
                return

            asset = self._find_zip_asset(data.get("assets", []))
            if asset is None:
                self.error.emit("Release 中未找到 Windows zip 包")
                return

            info = ReleaseInfo(
                tag=tag,
                name=data.get("name", tag),
                body=data.get("body", ""),
                download_url=asset["browser_download_url"],
                size=asset.get("size", 0),
            )
            self.finished.emit(info)
        except requests.RequestException as exc:
            self.error.emit(f"网络请求失败: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))

    @staticmethod
    def _find_zip_asset(assets: list) -> Optional[dict]:
        for a in assets:
            name: str = a.get("name", "")
            if name.endswith(".zip") and "windows" in name.lower():
                return a
        for a in assets:
            if a.get("name", "").endswith(".zip"):
                return a
        return None


class DownloadWorker(QThread):
    """后台线程：下载 zip 并解压到临时目录。"""

    progress = pyqtSignal(int)       # 0-100
    finished = pyqtSignal(str)       # 解压后的目录路径
    error = pyqtSignal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = _mirror_url(url)

    def run(self):
        try:
            resp = requests.get(self.url, stream=True, timeout=60)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            tmp_zip = os.path.join(tempfile.gettempdir(), "ez_traing_update.zip")

            downloaded = 0
            with open(tmp_zip, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        self.progress.emit(int(downloaded * 100 / total))

            self.progress.emit(100)

            extract_dir = os.path.join(tempfile.gettempdir(), "ez_traing_update")
            if os.path.isdir(extract_dir):
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)

            with zipfile.ZipFile(tmp_zip, "r") as zf:
                for member in zf.namelist():
                    member_path = os.path.realpath(os.path.join(extract_dir, member))
                    if not member_path.startswith(os.path.realpath(extract_dir) + os.sep) and member_path != os.path.realpath(extract_dir):
                        raise ValueError(f"Zip 包含非法路径: {member}")
                zf.extractall(extract_dir)

            os.remove(tmp_zip)
            self.finished.emit(extract_dir)
        except Exception as exc:
            self.error.emit(f"下载失败: {exc}")


def apply_update_and_restart(extracted_dir: str) -> None:
    """生成 bat 脚本执行文件替换，然后退出当前进程。

    仅在 frozen (PyInstaller) 环境下有效。
    """
    if not is_frozen():
        return

    exe_path = sys.executable
    app_dir = os.path.dirname(exe_path)
    exe_name = os.path.basename(exe_path)

    bat_path = os.path.join(app_dir, "_updater.bat")
    bat_content = f"""@echo off
chcp 65001 >nul
echo 正在更新，请稍候...
timeout /t 3 /nobreak >nul
xcopy /e /y /q "{extracted_dir}\\*" "{app_dir}\\"
if exist "{extracted_dir}" rmdir /s /q "{extracted_dir}"
start "" "{os.path.join(app_dir, exe_name)}"
del "%~f0"
"""
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    subprocess.Popen(
        ["cmd.exe", "/c", bat_path],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )

    from PyQt5.QtWidgets import QApplication
    QApplication.instance().quit()
