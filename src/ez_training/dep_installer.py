"""依赖安装模块 - 为 PyInstaller 发布版提供 torch/ultralytics 的安装能力。"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from ez_training.updater import is_frozen

ALIYUN_PYPI = "https://mirrors.aliyun.com/pypi/simple/"

TORCH_INDEX_URLS = {
    "CUDA 11.8": "https://download.pytorch.org/whl/cu118",
    "CUDA 12.1": "https://download.pytorch.org/whl/cu121",
    "CUDA 12.4": "https://download.pytorch.org/whl/cu124",
    "CPU": "https://download.pytorch.org/whl/cpu",
}

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def get_deps_dir() -> Path:
    """deps/ directory next to the exe (frozen) or project root (dev)."""
    if is_frozen():
        return Path(sys.executable).parent / "deps"
    return Path(__file__).resolve().parents[2] / "deps"


def find_system_python() -> Optional[str]:
    """Locate a working Python on the system PATH."""
    for name in ("py", "python", "python3"):
        try:
            r = subprocess.run(
                [name, "--version"],
                capture_output=True, text=True, timeout=10,
                creationflags=_CREATE_NO_WINDOW,
            )
            if r.returncode == 0 and "Python" in r.stdout + r.stderr:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def get_python_version(python_cmd: str) -> Optional[str]:
    """Return *major.minor* version string for *python_cmd*."""
    try:
        r = subprocess.run(
            [python_cmd, "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=10,
            creationflags=_CREATE_NO_WINDOW,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


class InstallWorker(QThread):
    """Background thread: install ultralytics + torch into deps/."""

    log = pyqtSignal(str)
    install_finished = pyqtSignal(bool, str)

    def __init__(self, cuda_variant: str, python_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.cuda_variant = cuda_variant
        self.python_path = python_path
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False

    def cancel(self):
        """Request cancellation and kill the running subprocess if any."""
        self._cancelled = True
        proc = self._process
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass

    def run(self):
        if self.python_path:
            python = self.python_path
        else:
            python = find_system_python()
        if python is None:
            self.install_finished.emit(
                False,
                "未找到系统 Python，请先安装 Python 3.10+ 并确保已加入 PATH",
            )
            return

        version = get_python_version(python)
        self.log.emit(f"使用 Python {version}  ({python})")

        if is_frozen() and version:
            bundled = f"{sys.version_info.major}.{sys.version_info.minor}"
            if version != bundled:
                self.install_finished.emit(
                    False,
                    f"Python 版本 ({version}) 与应用内置 Python ({bundled}) 不一致，"
                    f"安装的 C 扩展将无法加载。请提供 Python {bundled}.x 的路径后重试",
                )
                return

        deps_dir = get_deps_dir()
        deps_dir.mkdir(parents=True, exist_ok=True)
        self.log.emit(f"安装目标: {deps_dir}\n")

        self.log.emit("── 安装 ultralytics ──────────────────────")
        ok = self._run_pip(python, [
            "-m", "pip", "install",
            "--target", str(deps_dir),
            "-i", ALIYUN_PYPI,
            "--trusted-host", "mirrors.aliyun.com",
            "ultralytics",
        ])
        if self._cancelled:
            self.install_finished.emit(False, "安装已取消")
            return
        if not ok:
            hint = self._diagnose_failure(self._last_output)
            self.install_finished.emit(False, f"ultralytics 安装失败。{hint}")
            return

        self.log.emit("\n── 安装 PyTorch ──────────────────────────")
        torch_index = TORCH_INDEX_URLS.get(
            self.cuda_variant, TORCH_INDEX_URLS["CUDA 12.1"],
        )
        ok = self._run_pip(python, [
            "-m", "pip", "install",
            "--target", str(deps_dir),
            "torch", "torchvision", "torchaudio",
            "--index-url", torch_index,
        ])
        if self._cancelled:
            self.install_finished.emit(False, "安装已取消")
            return
        if not ok:
            hint = self._diagnose_failure(self._last_output)
            self.install_finished.emit(False, f"PyTorch 安装失败。{hint}")
            return

        self.install_finished.emit(True, "安装完成！重启应用后生效。")

    _last_output: str = ""

    def _run_pip(self, python: str, args: list) -> bool:
        cmd = [python, *args]
        self.log.emit(f"$ {' '.join(cmd)}\n")
        output_lines: list[str] = []
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
            )
            for line in self._process.stdout:
                if self._cancelled:
                    break
                stripped = line.rstrip()
                output_lines.append(stripped)
                self.log.emit(stripped)
            self._process.wait()
            ok = self._process.returncode == 0
            self._process = None
            self._last_output = "\n".join(output_lines[-50:])
            return ok
        except Exception as exc:
            self.log.emit(f"执行失败: {exc}")
            self._process = None
            self._last_output = str(exc)
            return False

    @staticmethod
    def _diagnose_failure(output: str) -> str:
        lowered = output.lower()
        if "no space left" in lowered or "disk" in lowered and "full" in lowered:
            return "磁盘空间不足，请清理后重试"
        if "connectionerror" in lowered or "timed out" in lowered or "connection" in lowered and "refused" in lowered:
            return "网络连接失败，请检查网络或代理设置"
        if "permission" in lowered and "denied" in lowered:
            return "权限不足，请尝试以管理员身份运行"
        if "no matching distribution" in lowered:
            return "未找到匹配的包版本，请检查 Python 版本和 CUDA 选项是否正确"
        if "could not find a version" in lowered:
            return "找不到合适的版本，可能是索引源不可用"
        return "请查看上方日志获取详细信息"
