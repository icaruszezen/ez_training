"""Reusable background workers shared across pages."""

import logging
import os
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QImageReader

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


class ThumbnailLoader(QThread):
    """Async thumbnail loader using QImage (thread-safe)."""

    thumbnail_loaded = pyqtSignal(str, QImage)
    all_loaded = pyqtSignal()

    def __init__(self, image_paths: List[str], thumbnail_size: int = 120):
        super().__init__()
        self.image_paths = image_paths
        self.thumbnail_size = thumbnail_size
        self._cancelled = False

    def run(self):
        for path in self.image_paths:
            if self._cancelled:
                break
            try:
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                orig_size = reader.size()
                if orig_size.isValid():
                    scaled_size = orig_size.scaled(
                        QSize(self.thumbnail_size, self.thumbnail_size),
                        Qt.KeepAspectRatio,
                    )
                    reader.setScaledSize(scaled_size)
                image = reader.read()
                if not image.isNull():
                    self.thumbnail_loaded.emit(path, image)
            except Exception:
                logger.debug("Failed to load thumbnail: %s", path, exc_info=True)
        if not self._cancelled:
            self.all_loaded.emit()

    def cancel(self):
        self._cancelled = True


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
                for root, dirs, files in os.walk(directory):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    if self._cancelled:
                        break
                    for f in files:
                        if Path(f).suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                            paths.append(os.path.join(root, f))
            paths.sort()
        except OSError as e:
            error = str(e)
        self.finished.emit(self._project_id, paths, error, perf_counter() - t0)
