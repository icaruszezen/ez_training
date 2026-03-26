"""应用自动更新模块。

通过 GitHub Releases API 检查新版本，下载 zip 并通过 bat 脚本完成替换重启。
"""

import hashlib
import os
import re
import sys
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional

import requests
from PyQt5.QtCore import QThread, pyqtSignal

from ez_training import __version__
from ez_training.common.constants import get_github_mirror_prefix

GITHUB_REPO = "icaruszezen/ez_training"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
REQUEST_TIMEOUT = 15


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 256), b""):
            h.update(block)
    return h.hexdigest()


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


_PRE_RELEASE_ORDER = {"alpha": 0, "a": 0, "beta": 1, "b": 1, "rc": 2}


def _compare_versions(remote: str, local: str) -> bool:
    """Return True if *remote* is strictly newer than *local*.

    Supports numeric versions (``0.2.1``) and pre-release suffixes
    like ``1.0.0-alpha.1``, ``1.0.0a2``, ``1.0.0-rc1``.  A release
    version is always newer than its pre-release counterparts.
    """
    def _parse(v: str):
        v = v.lstrip("vV")
        m = re.match(r"^(\d+(?:\.\d+)*)(?:[-.]?(alpha|beta|a|b|rc)\.?(\d*))?", v, re.I)
        if not m:
            raise ValueError(v)
        nums = tuple(int(x) for x in m.group(1).split("."))
        pre_tag = m.group(2)
        if pre_tag:
            pre_rank = _PRE_RELEASE_ORDER.get(pre_tag.lower(), -1)
            pre_num = int(m.group(3)) if m.group(3) else 0
            return (nums, 0, pre_rank, pre_num)
        return (nums, 1, 0, 0)
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
    sha256: str = ""


class CheckUpdateWorker(QThread):
    """后台线程：检查 GitHub 是否有新版本。"""

    finished = pyqtSignal(object)  # ReleaseInfo | None
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
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
            if self._cancelled:
                return
            data = resp.json()

            tag = data.get("tag_name", "")
            if not _compare_versions(tag, __version__):
                self.finished.emit(None)
                return

            asset = self._find_zip_asset(data.get("assets", []))
            if asset is None:
                self.error.emit("Release 中未找到 Windows zip 包")
                return

            body = data.get("body", "")
            sha256 = self._extract_sha256(body, asset.get("name", ""))

            info = ReleaseInfo(
                tag=tag,
                name=data.get("name", tag),
                body=body,
                download_url=asset["browser_download_url"],
                size=asset.get("size", 0),
                sha256=sha256,
            )
            if not self._cancelled:
                self.finished.emit(info)
        except requests.RequestException as exc:
            if not self._cancelled:
                self.error.emit(f"网络请求失败: {exc}")
        except Exception as exc:
            if not self._cancelled:
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

    @staticmethod
    def _extract_sha256(body: str, asset_name: str) -> str:
        """Extract SHA256 hash from release body.

        Recognises patterns like ``SHA256: <hex>`` or
        ``<hex>  <filename>`` (sha256sum output format).
        """
        if not body:
            return ""
        for line in body.splitlines():
            line = line.strip()
            m = re.match(r"(?i)sha-?256\s*[:=]\s*([0-9a-fA-F]{64})", line)
            if m:
                return m.group(1).lower()
            if asset_name:
                m = re.match(r"([0-9a-fA-F]{64})\s+" + re.escape(asset_name), line)
                if m:
                    return m.group(1).lower()
        return ""


class DownloadWorker(QThread):
    """后台线程：下载 zip 并解压到临时目录。"""

    progress = pyqtSignal(int)       # 0-100
    finished = pyqtSignal(str)       # 解压后的目录路径
    error = pyqtSignal(str)

    def __init__(self, url: str, expected_sha256: str = "", parent=None):
        super().__init__(parent)
        self._original_url = url
        self.url = _mirror_url(url)
        self._expected_sha256 = expected_sha256.lower()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        tmp_zip = os.path.join(tempfile.gettempdir(), "ez_training_update.zip")
        try:
            try:
                self._download(self.url, tmp_zip)
            except Exception:
                if self._cancelled:
                    raise
                if self.url != self._original_url:
                    self.progress.emit(0)
                    self._download(self._original_url, tmp_zip)
                else:
                    raise

            if self._cancelled:
                self._cleanup_file(tmp_zip)
                self.error.emit("下载已取消")
                return

            self.progress.emit(100)

            if self._expected_sha256:
                actual = _sha256_file(tmp_zip)
                if actual != self._expected_sha256:
                    raise ValueError(
                        f"校验和不匹配 (期望 {self._expected_sha256[:16]}…, "
                        f"实际 {actual[:16]}…)"
                    )

            extract_dir = os.path.join(tempfile.gettempdir(), "ez_training_update")
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
            self._cleanup_file(tmp_zip)
            if not self._cancelled:
                self.error.emit(f"下载失败: {exc}")

    def _download(self, url: str, dest: str):
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if self._cancelled:
                    return
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    self.progress.emit(int(downloaded * 100 / total))

    @staticmethod
    def _cleanup_file(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def apply_update_and_restart(extracted_dir: str) -> None:
    """生成 bat 脚本执行文件替换，然后退出当前进程。

    仅在 frozen (PyInstaller) 环境下有效。
    """
    if not is_frozen():
        return

    exe_path = sys.executable
    app_dir = os.path.dirname(exe_path)
    exe_name = os.path.basename(exe_path)

    backup_dir = os.path.join(tempfile.gettempdir(), "ez_training_backup")
    bat_path = os.path.join(app_dir, "_updater.bat")
    bat_content = f"""@echo off
chcp 65001 >nul
echo 正在更新，请稍候...
timeout /t 3 /nobreak >nul
if exist "{backup_dir}" rmdir /s /q "{backup_dir}"
mkdir "{backup_dir}"
echo 备份当前版本...
xcopy /e /y /q "{app_dir}\\*" "{backup_dir}\\" >nul 2>&1
echo 应用更新...
xcopy /e /y /q "{extracted_dir}\\*" "{app_dir}\\"
if errorlevel 1 (
    echo 更新失败，正在回滚...
    xcopy /e /y /q "{backup_dir}\\*" "{app_dir}\\" >nul 2>&1
    echo 已回滚到旧版本。
    pause
    goto :cleanup
)
:cleanup
if exist "{extracted_dir}" rmdir /s /q "{extracted_dir}"
if exist "{backup_dir}" rmdir /s /q "{backup_dir}"
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
