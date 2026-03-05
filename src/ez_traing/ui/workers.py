"""Reusable background workers shared across pages."""

import os
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS


class ThumbnailLoader(QThread):
    """Async thumbnail loader using QImage (thread-safe)."""

    thumbnail_loaded = pyqtSignal(str, QImage)
    all_loaded = pyqtSignal()

    def __init__(self, image_paths: List[str], thumbnail_size: int = 120):
        super().__init__()
        self.image_paths = image_paths
        self.thumbnail_size = thumbnail_size
        self._is_cancelled = False

    def run(self):
        for path in self.image_paths:
            if self._is_cancelled:
                break
            try:
                image = QImage(path)
                if not image.isNull():
                    scaled = image.scaled(
                        self.thumbnail_size,
                        self.thumbnail_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self.thumbnail_loaded.emit(path, scaled)
            except Exception:
                pass
        self.all_loaded.emit()

    def cancel(self):
        self._is_cancelled = True


class ImageScanWorker(QThread):
    """Scan one or more directory trees for images, keyed by project id."""

    finished = pyqtSignal(str, list, str, float)  # project_id, paths, error, elapsed_sec

    def __init__(self, project_id: str, directory: str = "",
                 directories: List[str] = None):
        super().__init__()
        self._project_id = project_id
        if directories:
            self._directories = list(directories)
        else:
            self._directories = [directory] if directory else []
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from time import perf_counter

        t0 = perf_counter()
        paths: List[str] = []
        error = ""
        try:
            for directory in self._directories:
                if self._cancelled:
                    break
                for root, _, files in os.walk(directory):
                    if self._cancelled:
                        break
                    for f in files:
                        if Path(f).suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                            paths.append(os.path.join(root, f))
            paths.sort()
        except OSError as e:
            error = str(e)
        self.finished.emit(self._project_id, paths, error, perf_counter() - t0)
