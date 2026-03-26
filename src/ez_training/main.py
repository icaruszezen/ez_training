import argparse
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if getattr(sys, "frozen", False):
    _deps_dir = Path(sys.executable).parent / "deps"
    if _deps_dir.is_dir():
        _deps_str = str(_deps_dir)

        import importlib
        import importlib.util

        class _DepsFinder:
            """Resolve packages excluded from the PyInstaller bundle from deps/.

            PyInstaller's FrozenImporter may not fall through to PathFinder for
            packages listed in ``excludes``.  This finder is inserted at the head
            of ``sys.meta_path`` so it intercepts those packages first and loads
            them directly from the deps/ directory.  Sub-module resolution is
            delegated back to the standard machinery via ``__path__``.
            """

            _PACKAGES = frozenset({
                "torch", "torchvision", "torchaudio", "ultralytics",
            })

            def find_spec(self, fullname, path, target=None):
                top = fullname.split(".", 1)[0]
                if top not in self._PACKAGES:
                    return None
                if "." in fullname:
                    return None
                pkg_dir = os.path.join(_deps_str, fullname)
                init_py = os.path.join(pkg_dir, "__init__.py")
                if os.path.isdir(pkg_dir) and os.path.isfile(init_py):
                    return importlib.util.spec_from_file_location(
                        fullname, init_py,
                        submodule_search_locations=[pkg_dir],
                    )
                single_py = os.path.join(_deps_str, fullname + ".py")
                if os.path.isfile(single_py):
                    return importlib.util.spec_from_file_location(fullname, single_py)
                return None

        sys.meta_path.insert(0, _DepsFinder())

        if _deps_str not in sys.path:
            sys.path.append(_deps_str)
        importlib.invalidate_caches()

        def _add_dll_dir(p: Path):
            if not p.is_dir():
                return
            s = str(p)
            try:
                os.add_dll_directory(s)
            except (OSError, AttributeError):
                pass
            if s not in os.environ.get("PATH", ""):
                os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")

        _add_dll_dir(_deps_dir / "torch" / "lib")
        _add_dll_dir(_deps_dir / "torch" / "bin")
        for _nv_lib in _deps_dir.glob("nvidia/*/lib"):
            _add_dll_dir(_nv_lib)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from ez_training.ui.main_window import AppWindow


def parse_args(argv):
    parser = argparse.ArgumentParser(description="EZ Training：Fluent 风格目标检测标注与训练工具")
    parser.add_argument("--smoke-test", action="store_true", help="启动后短时间内退出")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or [])

    app = QApplication(sys.argv)
    app.setApplicationName("EZ Training")
    setTheme(Theme.LIGHT)

    window = AppWindow()
    window.show()

    if args.smoke_test:
        QTimer.singleShot(800, app.quit)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
