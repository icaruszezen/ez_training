"""后台模板匹配工作线程，批量处理图片并发射进度信号。"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PyQt5.QtCore import QThread, pyqtSignal

from ez_traing.template_matching.matcher import MatchResult, TemplateMatcher, TemplateInfo

logger = logging.getLogger(__name__)


@dataclass
class TemplateMatchingStats:
    """批量匹配统计。"""

    total: int = 0
    processed: int = 0
    matched: int = 0
    empty: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: bool = False


class TemplateMatchingWorker(QThread):
    """批量模板匹配后台线程。

    Signals
    -------
    progress(int, int, str)
        (current, total, message)
    image_completed(str, bool, str, list)
        (image_path, success, message, boxes)
    finished(object)
        TemplateMatchingStats
    """

    progress = pyqtSignal(int, int, str)
    image_completed = pyqtSignal(str, bool, str, list)
    finished = pyqtSignal(object)

    def __init__(
        self,
        image_paths: List[str],
        templates: List[TemplateInfo],
        matcher: TemplateMatcher,
        skip_annotated: bool = True,
    ):
        super().__init__()
        self._image_paths = image_paths
        self._templates = templates
        self._matcher = matcher
        self._skip_annotated = skip_annotated
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        stats = TemplateMatchingStats(total=len(self._image_paths))
        tpl_cache = self._matcher.preprocess_templates(self._templates)

        for i, image_path in enumerate(self._image_paths):
            if self._cancelled:
                logger.info("模板匹配已被用户取消")
                stats.cancelled = True
                break

            filename = Path(image_path).name

            if self._skip_annotated and self._has_annotation(image_path):
                stats.skipped += 1
                msg = f"已有标注，跳过: {filename}"
                self.progress.emit(i + 1, stats.total, msg)
                self.image_completed.emit(image_path, True, msg, [])
                continue

            self.progress.emit(i + 1, stats.total, f"正在匹配: {filename}")

            try:
                result: MatchResult = self._matcher.match(
                    image_path, self._templates, _preprocessed_templates=tpl_cache
                )
            except Exception as exc:
                stats.processed += 1
                stats.failed += 1
                msg = f"匹配异常: {filename} - {exc}"
                logger.exception(msg)
                self.image_completed.emit(image_path, False, msg, [])
                continue

            stats.processed += 1

            if not result.success:
                stats.failed += 1
                self.image_completed.emit(
                    image_path, False, result.error_message, []
                )
                continue

            if result.boxes:
                stats.matched += 1
                msg = f"匹配完成: {filename} ({len(result.boxes)} 个候选框)"
            else:
                stats.empty += 1
                msg = f"未找到匹配: {filename}"

            self.image_completed.emit(image_path, True, msg, result.boxes)

        self.finished.emit(stats)

    @staticmethod
    def _has_annotation(image_path: str) -> bool:
        return Path(image_path).with_suffix(".xml").exists()
