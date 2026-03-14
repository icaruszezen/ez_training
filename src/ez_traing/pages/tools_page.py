"""小工具集合页面。"""

import os
import shutil
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CaptionLabel,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    ScrollArea,
    SubtitleLabel,
    TextEdit,
    TitleLabel,
)

from ez_traing.annotation_scripts.voc_utils import append_object, create_voc_root
from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS


class _YoloToVocWorker(QThread):
    """YOLO TXT -> VOC XML 后台转换线程。"""

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        image_dir: str,
        classes_file: str,
        label_dir: str,
        output_dir: str,
    ):
        super().__init__()
        self._image_dir = image_dir
        self._classes_file = classes_file
        self._label_dir = label_dir
        self._output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._do_convert()
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, f"转换失败: {e}")

    def _do_convert(self):
        from PIL import Image

        self.log.emit("读取类别文件...")
        classes, cls_warnings = self._read_classes(self._classes_file)
        if not classes:
            self.finished.emit(False, "classes.txt 为空或读取失败")
            return
        for w in cls_warnings:
            self.log.emit(f"[警告] {w}")
        self.log.emit(f"共 {len(classes)} 个类别: {', '.join(classes)}")

        self.log.emit(f"扫描图片目录: {self._image_dir}")
        image_base = Path(self._image_dir)
        image_files = sorted(
            f
            for f in image_base.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        )
        if not image_files:
            self.finished.emit(False, "未找到任何图片文件")
            return
        self.log.emit(f"找到 {len(image_files)} 张图片")

        self.log.emit(f"扫描标注目录: {self._label_dir}")
        label_map, lbl_warnings = self._build_label_map(self._label_dir)
        for w in lbl_warnings:
            self.log.emit(f"[警告] {w}")
        self.log.emit(f"找到 {len(label_map)} 个标注文件")

        output_path = Path(self._output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0
        errors = 0
        total = len(image_files)

        for i, img_path in enumerate(image_files):
            if self._cancelled:
                self.finished.emit(False, "已取消")
                return

            self.progress.emit(i + 1, total)
            stem = img_path.stem

            label_path = label_map.get(stem)
            if label_path is None:
                self.log.emit(f"[{i + 1}/{total}] {img_path.name} - 未找到标注，跳过")
                skipped += 1
                continue

            try:
                with Image.open(img_path) as im:
                    width, height = im.size
                    depth = len(im.getbands())

                boxes, skipped_lines = self._parse_yolo_txt(
                    label_path, classes, width, height
                )

                if not boxes:
                    tail = f" ({skipped_lines} 行被跳过)" if skipped_lines else ""
                    self.log.emit(
                        f"[{i + 1}/{total}] {img_path.name} - 标注为空，跳过"
                        + tail
                    )
                    skipped += 1
                    continue

                rel = img_path.relative_to(image_base)
                out_img = output_path / rel
                out_img.parent.mkdir(parents=True, exist_ok=True)

                root = create_voc_root(
                    folder=out_img.parent.name,
                    filename=img_path.name,
                    path=str(out_img),
                    width=width,
                    height=height,
                    depth=depth,
                )
                for label, xmin, ymin, xmax, ymax in boxes:
                    append_object(root, label, xmin, ymin, xmax, ymax)

                xml_out = out_img.with_suffix(".xml")
                tree = ET.ElementTree(root)
                try:
                    ET.indent(tree, space="    ")
                except AttributeError:
                    pass
                tree.write(str(xml_out), encoding="utf-8", xml_declaration=True)

                shutil.copy2(str(img_path), str(out_img))

                detail = f"[{i + 1}/{total}] {img_path.name} - {len(boxes)} 个目标"
                if skipped_lines:
                    detail += f" ({skipped_lines} 行被跳过)"
                self.log.emit(detail)
                converted += 1

            except Exception as e:
                self.log.emit(f"[{i + 1}/{total}] {img_path.name} - 错误: {e}")
                errors += 1

        msg = f"转换完成: 成功 {converted}, 跳过 {skipped}, 错误 {errors}, 共 {total} 张"
        self.log.emit(msg)
        self.finished.emit(True, msg)

    @staticmethod
    def _read_classes(path: str) -> Tuple[List[str], List[str]]:
        """返回 ``(classes, warnings)``。使用 utf-8-sig 自动去除 BOM。"""
        warnings: List[str] = []
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                classes = [line.strip() for line in f if line.strip()]
        except Exception:
            return [], []
        seen: Dict[str, int] = {}
        for i, name in enumerate(classes):
            if name in seen:
                warnings.append(
                    f"类别 '{name}' 重复 (行 {seen[name] + 1} 和 {i + 1})"
                )
            else:
                seen[name] = i
        return classes, warnings

    @staticmethod
    def _build_label_map(label_dir: str) -> Tuple[Dict[str, Path], List[str]]:
        """返回 ``(stem->path 映射, warnings)``，同名文件只保留首次出现的。"""
        result: Dict[str, Path] = {}
        warnings: List[str] = []
        for f in Path(label_dir).rglob("*.txt"):
            if f.name.lower() == "classes.txt":
                continue
            if f.stem in result:
                warnings.append(
                    f"标注 '{f.stem}.txt' 存在多个: 使用 {result[f.stem]}，忽略 {f}"
                )
            else:
                result[f.stem] = f
        return result, warnings

    @staticmethod
    def _parse_yolo_txt(
        txt_path: Path,
        classes: List[str],
        img_w: int,
        img_h: int,
    ) -> Tuple[List[Tuple[str, int, int, int, int]], int]:
        """返回 ``(boxes, skipped)``，*skipped* 为因格式/越界而跳过的行数。"""
        boxes: List[Tuple[str, int, int, int, int]] = []
        skipped = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 5:
                    skipped += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except (ValueError, IndexError):
                    skipped += 1
                    continue

                if cls_id < 0 or cls_id >= len(classes):
                    skipped += 1
                    continue

                xmin = max(0, round((xc - w / 2) * img_w))
                ymin = max(0, round((yc - h / 2) * img_h))
                xmax = min(img_w, round((xc + w / 2) * img_w))
                ymax = min(img_h, round((yc + h / 2) * img_h))

                if xmax > xmin and ymax > ymin:
                    boxes.append((classes[cls_id], xmin, ymin, xmax, ymax))
                else:
                    skipped += 1
        return boxes, skipped


class _YoloToVocCard(CardWidget):
    """YOLO TXT -> VOC XML 转换工具卡片。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[_YoloToVocWorker] = None
        self._setup_ui()
        self.destroyed.connect(self._stop_worker)

    def _stop_worker(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
            self._worker = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        layout.addWidget(SubtitleLabel("YOLO TXT → VOC XML 转换", self))
        layout.addWidget(
            CaptionLabel(
                "将 YOLO 格式的 .txt 标注文件转换为 Pascal VOC 格式的 .xml 标注文件，"
                "并将图片和标注输出到指定目录。",
                self,
            )
        )

        self.image_dir_edit = self._add_path_row(
            layout, "图片目录", "选择图片目录"
        )
        self.classes_file_edit = self._add_path_row(
            layout, "类别文件 (classes.txt)", "选择文件", is_file=True
        )
        self.label_dir_edit = self._add_path_row(
            layout, "标注文件目录", "选择标注目录"
        )
        self.output_dir_edit = self._add_path_row(
            layout, "输出目录", "选择输出目录"
        )

        btn_layout = QHBoxLayout()
        self.start_btn = PrimaryPushButton("开始转换", self)
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = PushButton("取消", self)
        self.cancel_btn.setIcon(FIF.CLOSE)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.log_edit = TextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(200)
        self.log_edit.setPlaceholderText("转换日志...")
        layout.addWidget(self.log_edit)

    def _add_path_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        btn_text: str,
        *,
        is_file: bool = False,
    ) -> LineEdit:
        parent_layout.addWidget(BodyLabel(label_text, self))
        row = QHBoxLayout()
        edit = LineEdit(self)
        edit.setPlaceholderText("请选择路径...")
        row.addWidget(edit, 1)
        btn = PushButton(btn_text, self)
        btn.setIcon(FIF.FOLDER)
        btn.clicked.connect(lambda _=None, e=edit, f=is_file: self._browse(e, f))
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return edit

    def _browse(self, edit: LineEdit, is_file: bool = False):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", "", "文本文件 (*.txt)"
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            edit.setText(path)

    def _on_start(self):
        image_dir = self.image_dir_edit.text().strip()
        classes_file = self.classes_file_edit.text().strip()
        label_dir = self.label_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not image_dir or not os.path.isdir(image_dir):
            InfoBar.error(
                "错误",
                "请选择有效的图片目录",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        if not classes_file or not os.path.isfile(classes_file):
            InfoBar.error(
                "错误",
                "请选择有效的 classes.txt 文件",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        if not label_dir or not os.path.isdir(label_dir):
            InfoBar.error(
                "错误",
                "请选择有效的标注文件目录",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        if not output_dir:
            InfoBar.error(
                "错误",
                "请选择输出目录",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        try:
            out_resolved = Path(output_dir).resolve()
            img_resolved = Path(image_dir).resolve()
            if out_resolved == img_resolved or img_resolved in out_resolved.parents:
                InfoBar.error(
                    "错误",
                    "输出目录不能与图片目录相同或是其父目录",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
                return
            out_resolved.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            InfoBar.error(
                "错误",
                f"输出目录无效或无法创建: {e}",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._worker = _YoloToVocWorker(image_dir, classes_file, label_dir, output_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _on_log(self, msg: str):
        self.log_edit.append(msg)
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_edit.setTextCursor(cursor)

    def _on_finished(self, success: bool, message: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if success:
            InfoBar.success(
                "完成",
                message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            InfoBar.error(
                "失败",
                message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        self._worker = None


class ToolsPage(QWidget):
    """小工具集合页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(36, 20, 36, 20)
        content_layout.setSpacing(16)

        content_layout.addWidget(TitleLabel("小工具", self))
        content_layout.addWidget(_YoloToVocCard(self))
        content_layout.addStretch()

        scroll_area.setWidget(content)
        main_layout.addWidget(scroll_area)
