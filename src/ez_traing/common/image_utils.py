"""Shared image I/O utilities with Unicode path support."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from a path that may contain non-ASCII characters (e.g. CJK).

    Uses ``np.fromfile`` + ``cv2.imdecode`` to bypass ``cv2.imread``'s
    ASCII-only limitation on Windows.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        logger.warning("无法读取图片文件: %s", path, exc_info=True)
        return None
