"""
批量标注页面
功能：多选图片，对第一张标注后将标注应用到所有分辨率一致的图片
"""

import logging
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QColor, QBrush, QKeySequence
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QAbstractItemView,
    QMessageBox,
    QStyledItemDelegate,
    QMenu,
    QAction,
    QShortcut,
)
from qfluentwidgets import (
    PushButton,
    PrimaryPushButton,
    TransparentPushButton,
    CardWidget,
    SubtitleLabel,
    TitleLabel,
    CaptionLabel,
    StrongBodyLabel,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    ProgressBar,
    ComboBox,
)

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS, load_settings
from ez_traing.labeling.annotation_window import AnnotationWindow
from ez_traing.labeling.label_file import LabelFile, LabelFileFormat
from ez_traing.ui.workers import ImageScanWorker, ThumbnailLoader


logger = logging.getLogger(__name__)

_COLOR_MISMATCH = QColor(255, 210, 210)
_COLOR_SUCCESS = QColor(210, 255, 210)
_COLOR_SAMPLED = QColor(255, 200, 50)


def _read_image_size(path: str) -> Optional[Tuple[int, int]]:
    """Read image dimensions without fully decoding the pixel data.

    Handles EXIF orientation so that reported width/height matches the
    visually displayed orientation.  Returns ``None`` when the file
    cannot be read.
    """
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if not size.isValid():
        return None
    return (size.width(), size.height())


def _shape_bbox_key(shape: dict) -> tuple:
    """用标签+包围盒坐标生成可比较的 key，用于识别同一标注。

    Uses ``round()`` instead of ``int()`` to avoid truncation bias when
    coordinates contain floating-point fractions (common after YOLO
    coordinate conversion).
    """
    points = shape["points"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (shape["label"], round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))


def _read_existing_voc_shapes(xml_path: str) -> Optional[List[dict]]:
    """读取已有 VOC XML 标注，转换为 shape dict 列表。

    Returns None on parse error (caller should skip the image to avoid data loss).
    Returns [] when the annotation file does not exist.
    """
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        shapes = []
        for obj in tree.getroot().findall("object"):
            name = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if not name or bnd is None:
                continue
            xmin = int(float((bnd.findtext("xmin") or "0").strip()))
            ymin = int(float((bnd.findtext("ymin") or "0").strip()))
            xmax = int(float((bnd.findtext("xmax") or "0").strip()))
            ymax = int(float((bnd.findtext("ymax") or "0").strip()))
            difficult = int((obj.findtext("difficult") or "0").strip())
            shapes.append({
                "label": name,
                "points": [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
                "difficult": bool(difficult),
            })
        return shapes
    except Exception as e:
        logger.warning("Failed to parse VOC annotation %s: %s", xml_path, e)
        return None


def _read_existing_yolo_shapes(
    txt_path: str, img_width: int, img_height: int, class_list: List[str]
) -> Optional[List[dict]]:
    """读取已有 YOLO TXT 标注，转换为 shape dict 列表。

    Returns None on parse error (caller should skip the image to avoid data loss).
    Returns [] when the annotation file does not exist.
    """
    from ez_traing.common.annotation_utils import read_yolo_boxes

    if not os.path.exists(txt_path):
        return []
    try:
        boxes = read_yolo_boxes(Path(txt_path), img_width, img_height, class_list)
        shapes = []
        for b in boxes:
            xmin, ymin = b["xmin"], b["ymin"]
            xmax, ymax = b["xmax"], b["ymax"]
            shapes.append({
                "label": b["label"],
                "points": [
                    (xmin, ymin), (xmax, ymin),
                    (xmax, ymax), (xmin, ymax),
                ],
                "difficult": False,
            })
        return shapes
    except Exception as e:
        logger.warning("Failed to parse YOLO annotation %s: %s", txt_path, e)
        return None


class BatchApplyWorker(QThread):
    """后台批量应用标注到多张图片"""

    progress = pyqtSignal(int, int)  # current, total
    image_done = pyqtSignal(str, bool, str)  # path, success, message
    finished = pyqtSignal(int, int, int, list)  # applied, skipped, failed, mismatch_paths

    def __init__(
        self,
        shapes: List[dict],
        ref_size: Tuple[int, int],
        target_paths: List[str],
        label_format: LabelFileFormat,
        class_list: List[str],
    ):
        super().__init__()
        self._shapes = shapes
        self._ref_size = ref_size
        self._target_paths = target_paths
        self._label_format = label_format
        self._class_list = class_list
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        applied = 0
        skipped = 0
        failed = 0
        mismatch_paths: List[str] = []
        total = len(self._target_paths)

        new_shapes_data = [
            {
                "label": s["label"],
                "points": s["points"],
                "difficult": s.get("difficult", False),
            }
            for s in self._shapes
        ]
        for i, path in enumerate(self._target_paths):
            if self._cancelled:
                break
            try:
                img_size = _read_image_size(path)
                if img_size is None:
                    self.image_done.emit(path, False, f"无法读取: {Path(path).name}")
                    failed += 1
                    self.progress.emit(i + 1, total)
                    continue

                if img_size != self._ref_size:
                    mismatch_paths.append(path)
                    self.image_done.emit(
                        path,
                        False,
                        f"分辨率不匹配: {Path(path).name} "
                        f"({img_size[0]}x{img_size[1]} != "
                        f"{self._ref_size[0]}x{self._ref_size[1]})",
                    )
                    skipped += 1
                    self.progress.emit(i + 1, total)
                    continue

                p = Path(path)

                # 读取目标图片已有标注
                if self._label_format == LabelFileFormat.YOLO:
                    ann_path = str(p.with_suffix(".txt"))
                    existing = _read_existing_yolo_shapes(
                        ann_path, img_size[0], img_size[1], self._class_list
                    )
                else:
                    ann_path = str(p.with_suffix(".xml"))
                    existing = _read_existing_voc_shapes(ann_path)

                if existing is None:
                    self.image_done.emit(
                        path, False,
                        f"标注文件损坏，跳过: {Path(path).name}",
                    )
                    failed += 1
                    self.progress.emit(i + 1, total)
                    continue

                # 去重合并：保留已有标注 + 追加新标注中不重复的
                existing_keys = {_shape_bbox_key(s) for s in existing}
                to_add = [s for s in new_shapes_data if _shape_bbox_key(s) not in existing_keys]
                merged = existing + to_add

                label_file = LabelFile()
                if self._label_format == LabelFileFormat.YOLO:
                    save_class_list = list(self._class_list)
                    for s in merged:
                        if s["label"] not in save_class_list:
                            save_class_list.append(s["label"])
                    out = str(p.with_suffix(".txt"))
                    label_file.save_yolo_format(
                        out, merged, path, None, save_class_list
                    )
                else:
                    out = str(p.with_suffix(".xml"))
                    label_file.save_pascal_voc_format(out, merged, path, None)

                added_count = len(to_add)
                kept_count = len(existing)
                self.image_done.emit(
                    path, True,
                    f"已保存: {Path(out).name} (保留{kept_count}个, 新增{added_count}个)",
                )
                applied += 1

            except Exception as e:
                self.image_done.emit(path, False, f"错误 {Path(path).name}: {e}")
                failed += 1

            self.progress.emit(i + 1, total)

        self.finished.emit(applied, skipped, failed, mismatch_paths)


class _ColorItemDelegate(QStyledItemDelegate):
    """在 stylesheet 控制下仍能正确显示 item 背景色的 delegate。

    Qt stylesheet 会完全接管 item 绘制，导致 setBackground() 失效。
    此 delegate 在 stylesheet 绘制之前先画出 BackgroundRole 中的颜色。
    """

    def paint(self, painter, option, index):
        bg = index.data(Qt.BackgroundRole)
        if isinstance(bg, QBrush) and bg.color().alpha() > 0:
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(option.rect.adjusted(1, 1, -1, -1), 4, 4)
            painter.restore()
        super().paint(painter, option, index)


class BatchImageListPanel(CardWidget):
    """右侧多选图片列表面板"""

    selection_changed = pyqtSignal()
    first_image_changed = pyqtSignal(str)  # 第一张选中图片路径变化

    sample_mark_requested = pyqtSignal(str, bool)  # path, mark (True=add, False=remove)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
        self._all_paths: List[str] = []
        self._path_to_item: Dict[str, QListWidgetItem] = {}
        self._thumbnail_cache: Dict[str, QPixmap] = {}
        self._mismatch_paths: set = set()
        self._success_paths: set = set()
        self._sampled_paths: Set[str] = set()
        self._thumbnail_loader: Optional[ThumbnailLoader] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title_label = SubtitleLabel("图片列表")
        layout.addWidget(self.title_label)

        self.count_label = CaptionLabel("共 0 张图片，已选 0 张")
        layout.addWidget(self.count_label)

        self.ref_info_label = CaptionLabel("")
        layout.addWidget(self.ref_info_label)

        # 图片列表
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setIconSize(QSize(60, 60))
        self.image_list.setSpacing(2)
        self.image_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                outline: none;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 4px;
                background-color: transparent;
            }
            QListWidget::item:selected {
                background-color: rgba(227, 242, 253, 200);
                border: 1px solid #2196f3;
            }
            QListWidget::item:hover {
                background-color: rgba(245, 245, 245, 180);
            }
        """)
        self.image_list.setItemDelegate(_ColorItemDelegate(self.image_list))
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._show_context_menu)
        self.image_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.image_list, 1)

        # 选择按钮行
        sel_layout = QHBoxLayout()
        sel_layout.setSpacing(6)
        self.select_all_btn = TransparentPushButton("全选", self)
        self.select_all_btn.setFixedHeight(30)
        self.select_all_btn.clicked.connect(self._select_all)
        sel_layout.addWidget(self.select_all_btn)

        self.invert_btn = TransparentPushButton("反选", self)
        self.invert_btn.setFixedHeight(30)
        self.invert_btn.clicked.connect(self._invert_selection)
        sel_layout.addWidget(self.invert_btn)

        self.clear_sel_btn = TransparentPushButton("取消选择", self)
        self.clear_sel_btn.setFixedHeight(30)
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        sel_layout.addWidget(self.clear_sel_btn)

        sel_layout.addStretch()
        layout.addLayout(sel_layout)

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def set_images(self, paths: List[str]):
        """设置图片列表"""
        self._stop_thumbnail_loader()
        self.image_list.clear()
        self._path_to_item.clear()
        self._all_paths = list(paths)
        self._mismatch_paths.clear()
        self._success_paths.clear()
        self._sampled_paths.clear()

        placeholder = QPixmap(60, 60)
        placeholder.fill(Qt.lightGray)

        for path in self._all_paths:
            item = QListWidgetItem()
            cached = self._thumbnail_cache.get(path)
            item.setIcon(QIcon(cached if cached else placeholder))
            item.setText(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            item.setSizeHint(QSize(0, 68))
            self.image_list.addItem(item)
            self._path_to_item[self._norm(path)] = item

        self._update_count_label()
        self._load_thumbnails()

    def _load_thumbnails(self):
        uncached = [p for p in self._all_paths if p not in self._thumbnail_cache]
        if not uncached:
            return
        self._thumbnail_loader = ThumbnailLoader(uncached)
        self._thumbnail_loader.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self._thumbnail_loader.start()

    def _on_thumbnail_loaded(self, path: str, image: QImage):
        pixmap = QPixmap.fromImage(image)
        self._thumbnail_cache[path] = pixmap
        item = self._path_to_item.get(self._norm(path))
        if item:
            item.setIcon(QIcon(pixmap))

    def _stop_thumbnail_loader(self):
        if self._thumbnail_loader and self._thumbnail_loader.isRunning():
            self._thumbnail_loader.cancel()
            self._thumbnail_loader.wait()
        self._thumbnail_loader = None

    def mark_mismatch(self, paths: List[str]):
        """将分辨率不匹配的图片标红"""
        for path in paths:
            norm = self._norm(path)
            self._mismatch_paths.add(norm)
            item = self._path_to_item.get(norm)
            if item:
                item.setBackground(QBrush(_COLOR_MISMATCH))
                current = item.text()
                if "[分辨率不匹配]" not in current:
                    item.setText(f"{current}  [分辨率不匹配]")

    def mark_success(self, path: str):
        """将成功标注的图片标绿"""
        norm = self._norm(path)
        if norm in self._mismatch_paths:
            return
        self._success_paths.add(norm)
        item = self._path_to_item.get(norm)
        if item:
            item.setBackground(QBrush(_COLOR_SUCCESS))

    def clear_marks(self, keep_sampled: bool = True):
        """清除所有标记（加样标记可选保留）"""
        self._mismatch_paths.clear()
        self._success_paths.clear()
        for item in self._path_to_item.values():
            original_path = item.data(Qt.UserRole)
            norm = self._norm(original_path) if original_path else ""
            if keep_sampled and norm in self._sampled_paths:
                item.setBackground(QBrush(_COLOR_SAMPLED))
                item.setText(f"{Path(original_path).name}  [加样]")
            else:
                item.setBackground(QBrush())
                if original_path:
                    item.setText(Path(original_path).name)
        if not keep_sampled:
            self._sampled_paths.clear()

    def mark_sampled(self, path: str):
        """标记图片为加样状态"""
        norm = self._norm(path)
        self._sampled_paths.add(norm)
        item = self._path_to_item.get(norm)
        if item:
            item.setBackground(QBrush(_COLOR_SAMPLED))
            original_path = item.data(Qt.UserRole)
            item.setText(f"{Path(original_path).name}  [加样]")

    def unmark_sampled(self, path: str):
        """取消图片的加样标记"""
        norm = self._norm(path)
        self._sampled_paths.discard(norm)
        item = self._path_to_item.get(norm)
        if item:
            item.setBackground(QBrush())
            original_path = item.data(Qt.UserRole)
            if original_path:
                item.setText(Path(original_path).name)

    def is_sampled(self, path: str) -> bool:
        return self._norm(path) in self._sampled_paths

    def _show_context_menu(self, pos):
        """右键菜单"""
        selected = self.image_list.selectedItems()
        if not selected:
            return

        menu = QMenu(self)
        count = len(selected)

        # 加样操作
        sampled_count = sum(
            1 for it in selected if self._norm(it.data(Qt.UserRole)) in self._sampled_paths
        )
        not_sampled_count = count - sampled_count

        if not_sampled_count > 0:
            act_sample = QAction(f"标记为加样 ({not_sampled_count} 张)", menu)
            act_sample.triggered.connect(self._request_mark_selected_sampled)
            menu.addAction(act_sample)

        if sampled_count > 0:
            act_unsample = QAction(f"取消加样 ({sampled_count} 张)", menu)
            act_unsample.triggered.connect(self._request_unmark_selected_sampled)
            menu.addAction(act_unsample)

        menu.addSeparator()

        act_green = QAction(f"标记为绿色 ({count} 张)", menu)
        act_green.triggered.connect(lambda: self._mark_selected_color(_COLOR_SUCCESS))
        menu.addAction(act_green)

        act_red = QAction(f"标记为红色 ({count} 张)", menu)
        act_red.triggered.connect(lambda: self._mark_selected_color(_COLOR_MISMATCH))
        menu.addAction(act_red)

        menu.addSeparator()

        act_clear = QAction(f"清除选中标记 ({count} 张)", menu)
        act_clear.triggered.connect(self._clear_selected_marks)
        menu.addAction(act_clear)

        act_clear_all = QAction("清除全部标记", menu)
        act_clear_all.triggered.connect(self.clear_marks)
        menu.addAction(act_clear_all)

        menu.exec_(self.image_list.viewport().mapToGlobal(pos))

    def _request_mark_selected_sampled(self):
        for item in self.image_list.selectedItems():
            path = item.data(Qt.UserRole)
            if not self.is_sampled(path):
                self.sample_mark_requested.emit(path, True)

    def _request_unmark_selected_sampled(self):
        for item in self.image_list.selectedItems():
            path = item.data(Qt.UserRole)
            if self.is_sampled(path):
                self.sample_mark_requested.emit(path, False)

    def _mark_selected_color(self, color: QColor):
        """将当前选中的图片标记为指定颜色"""
        for item in self.image_list.selectedItems():
            path = item.data(Qt.UserRole)
            norm = self._norm(path)
            item.setBackground(QBrush(color))
            if color == _COLOR_SUCCESS:
                self._success_paths.add(norm)
                self._mismatch_paths.discard(norm)
                if "[分辨率不匹配]" in item.text():
                    item.setText(Path(path).name)
            elif color == _COLOR_MISMATCH:
                self._mismatch_paths.add(norm)
                self._success_paths.discard(norm)

    def _clear_selected_marks(self):
        """清除当前选中图片的标记"""
        for item in self.image_list.selectedItems():
            path = item.data(Qt.UserRole)
            norm = self._norm(path)
            item.setBackground(QBrush())
            item.setText(Path(path).name)
            self._mismatch_paths.discard(norm)
            self._success_paths.discard(norm)

    def get_selected_paths(self) -> List[str]:
        """获取选中图片路径"""
        return [
            item.data(Qt.UserRole)
            for item in self.image_list.selectedItems()
        ]

    def get_first_selected_path(self) -> Optional[str]:
        selected = self.image_list.selectedItems()
        if selected:
            return selected[0].data(Qt.UserRole)
        return None

    def _on_selection_changed(self):
        self._update_count_label()
        self.selection_changed.emit()
        first = self.get_first_selected_path()
        if first:
            self.first_image_changed.emit(first)

    def _update_count_label(self):
        total = len(self._all_paths)
        selected = len(self.image_list.selectedItems())
        self.count_label.setText(f"共 {total} 张图片，已选 {selected} 张")

    def _select_all(self):
        self.image_list.selectAll()

    def _invert_selection(self):
        self.image_list.blockSignals(True)
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            item.setSelected(not item.isSelected())
        self.image_list.blockSignals(False)
        self._on_selection_changed()

    def _clear_selection(self):
        self.image_list.clearSelection()

    def select_paths(self, paths: List[str]):
        """编程方式选中指定路径"""
        self.image_list.clearSelection()
        for path in paths:
            item = self._path_to_item.get(self._norm(path))
            if item:
                item.setSelected(True)

    def highlight_path(self, path: str):
        """高亮并滚动到指定路径，不触发 first_image_changed 信号"""
        item = self._path_to_item.get(self._norm(path))
        if not item:
            return
        self.image_list.blockSignals(True)
        self.image_list.clearSelection()
        item.setSelected(True)
        self.image_list.scrollToItem(item)
        self.image_list.blockSignals(False)
        self._update_count_label()


class BatchAnnotationPage(QWidget):
    """批量标注页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = None
        self._project_ids: List[str] = []
        self._current_directory: Optional[str] = None
        self._worker: Optional[BatchApplyWorker] = None
        self._scan_worker: Optional[ImageScanWorker] = None
        self._annotation_window: Optional[AnnotationWindow] = None
        self._baseline_shapes: List[dict] = []
        # 加样状态
        self._sampled_paths: Set[str] = set()
        self._sample_dataset_dir: Optional[str] = None
        self._sample_name_map: Dict[str, str] = {}  # norm(src) -> target filename
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部工具栏
        header = self._create_header()
        main_layout.addWidget(header)

        # 进度条
        self.progress_bar = ProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(4)
        main_layout.addWidget(self.progress_bar)

        # 主区域: 左标注 + 右列表
        content_splitter = QSplitter(Qt.Horizontal)

        # 左侧: 标注窗口
        self._annotation_window = AnnotationWindow(parent=self)
        self._syncing_selection = False
        self._annotation_window.file_loaded.connect(self._after_annotation_load)
        self._annotation_window.file_saved.connect(self._on_file_saved)
        content_splitter.addWidget(self._annotation_window)

        # 右侧: 图片列表面板
        self.image_panel = BatchImageListPanel()
        self.image_panel.first_image_changed.connect(self._on_first_image_changed)
        self.image_panel.sample_mark_requested.connect(self._on_sample_mark_from_panel)
        content_splitter.addWidget(self.image_panel)

        content_splitter.setSizes([800, 320])
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(content_splitter, 1)

        # 底部状态栏
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(12, 4, 12, 4)
        self.status_label = CaptionLabel("就绪")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)

        # 快捷键
        sc = QShortcut(QKeySequence("G"), self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self._toggle_sample_mark)

    def _create_header(self) -> CardWidget:
        header = CardWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        title = TitleLabel("批量标注")
        layout.addWidget(title)

        layout.addSpacing(16)

        # 数据集选择
        layout.addWidget(StrongBodyLabel("数据集:"))
        self.dataset_combo = ComboBox()
        self.dataset_combo.setMinimumWidth(200)
        self.dataset_combo.setPlaceholderText("选择数据集项目")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        layout.addWidget(self.dataset_combo)

        layout.addStretch()

        # 操作按钮
        self.sample_btn = PushButton("标记加样 (G)", self, FIF.FLAG)
        self.sample_btn.clicked.connect(self._toggle_sample_mark)
        layout.addWidget(self.sample_btn)

        self.apply_btn = PrimaryPushButton("应用标注到选中图片", self, FIF.SEND)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_btn)

        self.clear_marks_btn = PushButton("清除标记", self, FIF.SYNC)
        self.clear_marks_btn.clicked.connect(self._on_clear_marks)
        layout.addWidget(self.clear_marks_btn)

        return header

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project_manager(self, manager):
        self._project_manager = manager

    def load_images(self, directory: str, paths: List[str]):
        """从外部加载指定图片（由数据集页批量标注按钮调用）"""
        self._current_directory = directory
        self._reset_sample_state()
        self.image_panel.set_images(paths)

        if paths:
            self.image_panel.select_paths(paths)
            self._load_first_image(paths[0], directory)
            self.status_label.setText(f"已加载 {len(paths)} 张图片")
        else:
            self.status_label.setText("未选择图片")

        self._sync_dataset_combo_to_directory(directory)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_dataset_list()

    # ------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------

    def _refresh_dataset_list(self):
        if self._project_manager is None:
            return

        prev_id = None
        if self._project_ids and self.dataset_combo.currentIndex() >= 0:
            idx = self.dataset_combo.currentIndex()
            if idx < len(self._project_ids):
                prev_id = self._project_ids[idx]

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self._project_ids.clear()

        for proj in self._project_manager.get_all_projects(exclude_archived=True):
            self.dataset_combo.addItem(f"{proj.name} ({proj.image_count} 张)")
            self._project_ids.append(proj.id)

        if prev_id and prev_id in self._project_ids:
            self.dataset_combo.setCurrentIndex(self._project_ids.index(prev_id))
            self.dataset_combo.blockSignals(False)
        elif self._project_ids:
            self.dataset_combo.setCurrentIndex(0)
            self.dataset_combo.blockSignals(False)
            self._on_dataset_changed(0)
        else:
            self.dataset_combo.blockSignals(False)

    def _sync_dataset_combo_to_directory(self, directory: str):
        """让 ComboBox 选中对应目录的项目"""
        if not self._project_manager:
            return
        for i, pid in enumerate(self._project_ids):
            proj = self._project_manager.get_project(pid)
            if proj and proj.directory == directory:
                self.dataset_combo.blockSignals(True)
                self.dataset_combo.setCurrentIndex(i)
                self.dataset_combo.blockSignals(False)
                return

    def _on_dataset_changed(self, index: int):
        if index < 0 or index >= len(self._project_ids):
            return
        proj = self._project_manager.get_project(self._project_ids[index])
        if not proj:
            return

        dirs = self._project_manager.get_directories(proj.id)
        if not dirs:
            if not proj.is_archive_root:
                InfoBar.error(
                    title="错误",
                    content=f"目录不存在: {proj.directory}",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
            return

        self._current_directory = dirs[0]
        self._reset_sample_state()

        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.finished.disconnect(self._on_scan_finished)
            self._scan_worker.cancel()
            self._scan_worker.wait()

        self.status_label.setText(f"正在扫描: {proj.name}...")
        self._scan_worker = ImageScanWorker(proj.id, directories=dirs)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_finished(self, _project_id: str, paths: List[str], _error: str, _elapsed: float):
        self.image_panel.set_images(paths)
        self.status_label.setText(f"已加载 {len(paths)} 张图片")

        if paths:
            self._load_first_image(paths[0], self._current_directory)

    # ------------------------------------------------------------------
    # Annotation window interaction
    # ------------------------------------------------------------------

    def _load_first_image(self, image_path: str, directory: Optional[str] = None):
        if not self._annotation_window or not os.path.exists(image_path):
            return

        if directory and os.path.isdir(directory):
            current_dir = getattr(self._annotation_window, "dir_name", None)
            if current_dir != directory:
                self._annotation_window.import_dir_images(directory)
                self._annotation_window.default_save_dir = directory

        self._annotation_window.load_file(image_path)

        # 同步 cur_img_idx，否则按 d/a 切换时会从旧位置跳
        abs_path = os.path.abspath(image_path)
        m_img_list = getattr(self._annotation_window, "m_img_list", [])
        if abs_path in m_img_list:
            self._annotation_window.cur_img_idx = m_img_list.index(abs_path)

        # 记录加载时已有的标注作为基线，后续只应用新增的标注
        self._baseline_shapes = self._annotation_window._snapshot_current_shapes()

        img_size = _read_image_size(image_path)
        if img_size is not None:
            baseline_count = len(self._baseline_shapes)
            info = f"参考分辨率: {img_size[0]} x {img_size[1]}"
            if baseline_count:
                info += f" | 已有 {baseline_count} 个标注"
            self.image_panel.ref_info_label.setText(info)

    def _on_first_image_changed(self, path: str):
        """当右侧列表中第一张选中图片变化时，加载到标注窗口"""
        if self._syncing_selection:
            return
        self._syncing_selection = True
        self._load_first_image(path, self._current_directory)
        self._syncing_selection = False

    def _after_annotation_load(self):
        """AnnotationWindow 加载完一张图片后，同步右侧面板"""
        if self._syncing_selection:
            return
        file_path = getattr(self._annotation_window, "file_path", None)
        if not file_path:
            return
        file_path = os.path.abspath(file_path)
        self._syncing_selection = True
        self.image_panel.highlight_path(file_path)
        self._baseline_shapes = self._annotation_window._snapshot_current_shapes()
        img_size = _read_image_size(file_path)
        if img_size is not None:
            baseline_count = len(self._baseline_shapes)
            info = f"参考分辨率: {img_size[0]} x {img_size[1]}"
            if baseline_count:
                info += f" | 已有 {baseline_count} 个标注"
            self.image_panel.ref_info_label.setText(info)
        self._syncing_selection = False

    # ------------------------------------------------------------------
    # Batch apply
    # ------------------------------------------------------------------

    def _on_apply_clicked(self):
        if not self._annotation_window:
            return

        # 获取当前所有标注
        all_shapes = self._annotation_window._snapshot_current_shapes()
        if not all_shapes:
            InfoBar.warning(
                title="提示",
                content="当前图片没有标注框，请先标注",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        # 与基线对比，只提取新增的标注
        baseline_keys = {_shape_bbox_key(s) for s in self._baseline_shapes}
        new_shapes = [s for s in all_shapes if _shape_bbox_key(s) not in baseline_keys]

        if not new_shapes:
            InfoBar.warning(
                title="提示",
                content="没有检测到新增的标注框（与加载时相同），请先添加新标注",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        # 获取选中图片
        selected = self.image_panel.get_selected_paths()
        if len(selected) < 2:
            InfoBar.warning(
                title="提示",
                content="请至少选择2张图片（第1张为模板，其余为目标）",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        ref_path = selected[0]
        target_paths = selected[1:]

        # 获取参考图分辨率
        ref_size = _read_image_size(ref_path)
        if ref_size is None:
            InfoBar.error(
                title="错误",
                content="无法读取参考图片",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        # 先保存当前标注
        if hasattr(self._annotation_window, "save_file"):
            self._annotation_window.save_file()

        # 获取标注格式和类别列表
        label_format = getattr(
            self._annotation_window,
            "label_file_format",
            LabelFileFormat.PASCAL_VOC,
        )
        class_list = list(getattr(self._annotation_window, "label_hist", []))

        if label_format == LabelFileFormat.YOLO and not class_list:
            InfoBar.error(
                title="错误",
                content="YOLO 格式需要类别列表，但当前为空。请先确保标注中包含至少一个类别。",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=4000,
            )
            return

        # 确认对话框
        reply = QMessageBox.question(
            self,
            "确认批量应用",
            f"将 {len(new_shapes)} 个新增标注框追加到 {len(target_paths)} 张图片？\n"
            f"（不影响目标图片已有的标注）\n"
            f"分辨率不匹配的图片将被跳过。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self._set_running_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._worker = BatchApplyWorker(
            shapes=new_shapes,
            ref_size=ref_size,
            target_paths=target_paths,
            label_format=label_format,
            class_list=class_list,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.status_label.setText(f"正在应用: {current}/{total}")

    def _on_image_done(self, path: str, success: bool, message: str):
        if success:
            self.image_panel.mark_success(path)

    def _on_finished(self, applied: int, skipped: int, failed: int, mismatch_paths: list):
        self._set_running_state(False)
        self.progress_bar.setValue(100)

        if mismatch_paths:
            self.image_panel.mark_mismatch(mismatch_paths)

        summary = f"完成: {applied} 张成功"
        if skipped:
            summary += f", {skipped} 张分辨率不匹配已跳过"
        if failed:
            summary += f", {failed} 张失败"
        self.status_label.setText(summary)

        if skipped > 0:
            InfoBar.warning(
                title="批量标注完成",
                content=f"{applied} 张成功，{skipped} 张因分辨率不匹配已跳过（已标红）",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            InfoBar.success(
                title="批量标注完成",
                content=f"全部 {applied} 张图片标注成功",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )

        self._worker = None

    def _on_clear_marks(self):
        self.image_panel.clear_marks()

    def _cancel_apply(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.status_label.setText("正在取消...")

    def _set_running_state(self, running: bool):
        self.dataset_combo.setEnabled(not running)
        self.clear_marks_btn.setEnabled(not running)
        self.sample_btn.setEnabled(not running)
        if running:
            self.apply_btn.setText("取消")
            self.apply_btn.setEnabled(True)
            try:
                self.apply_btn.clicked.disconnect(self._on_apply_clicked)
            except TypeError:
                pass
            self.apply_btn.clicked.connect(self._cancel_apply)
        else:
            try:
                self.apply_btn.clicked.disconnect(self._cancel_apply)
            except TypeError:
                pass
            self.apply_btn.setText("应用标注到选中图片")
            self.apply_btn.setEnabled(True)
            self.apply_btn.clicked.connect(self._on_apply_clicked)

    # ------------------------------------------------------------------
    # Sample mark (加样)
    # ------------------------------------------------------------------

    def _reset_sample_state(self):
        self._sampled_paths.clear()
        self._sample_dataset_dir = None
        self._sample_name_map.clear()
        self.image_panel.clear_marks(keep_sampled=False)

    def _get_current_dataset_name(self) -> Optional[str]:
        if not self._project_manager or not self._project_ids:
            return None
        idx = self.dataset_combo.currentIndex()
        if idx < 0 or idx >= len(self._project_ids):
            return None
        proj = self._project_manager.get_project(self._project_ids[idx])
        return proj.name if proj else None

    def _ensure_sample_dataset_dir(self) -> Optional[str]:
        """确保加样数据集目录已创建，返回路径或 None"""
        if self._sample_dataset_dir and os.path.isdir(self._sample_dataset_dir):
            return self._sample_dataset_dir

        settings = load_settings()
        root = settings.get("sample_dataset_dir", "").strip()
        if not root:
            InfoBar.warning(
                title="未配置加样目录",
                content="请先在「设置」页面配置加样数据集存放目录",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=4000,
            )
            return None

        dataset_name = self._get_current_dataset_name()
        if not dataset_name:
            InfoBar.warning(
                title="提示",
                content="请先选择数据集",
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return None

        target_dir = os.path.join(root, f"{dataset_name}_加样")
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            InfoBar.error(
                title="创建目录失败",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=4000,
            )
            return None

        self._sample_dataset_dir = target_dir

        if self._project_manager:
            already_exists = any(
                p.directory == target_dir
                for p in self._project_manager.get_all_projects()
            )
            if not already_exists:
                from ez_traing.pages.dataset_page import DatasetProject
                proj = DatasetProject.create(f"{dataset_name}_加样", target_dir)
                self._project_manager.add_project(proj)

        InfoBar.success(
            title="已创建加样数据集",
            content=f"目录: {target_dir}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000,
        )
        return target_dir

    def _annotation_ext(self) -> str:
        label_format = getattr(
            self._annotation_window,
            "label_file_format",
            LabelFileFormat.PASCAL_VOC,
        )
        return ".txt" if label_format == LabelFileFormat.YOLO else ".xml"

    def _resolve_sample_name(self, image_path: str) -> str:
        """Return the target filename for *image_path* inside the sample dir.

        Re-uses a previously assigned name for the same source.  When the
        preferred name is already taken by a *different* source, a numeric
        suffix ``_2``, ``_3``, ... is appended until a free slot is found.
        """
        norm = os.path.normcase(os.path.abspath(image_path))
        existing_name = self._sample_name_map.get(norm)
        if existing_name:
            return existing_name

        p = Path(image_path)
        candidate = p.name
        occupied_names = set(self._sample_name_map.values())

        if candidate in occupied_names:
            stem, suffix = p.stem, p.suffix
            counter = 2
            while f"{stem}_{counter}{suffix}" in occupied_names:
                counter += 1
            candidate = f"{stem}_{counter}{suffix}"

        self._sample_name_map[norm] = candidate
        return candidate

    def _copy_to_sample_dir(self, image_path: str) -> bool:
        """复制图片及标注文件到加样数据集目录，返回图片是否复制成功"""
        if not self._sample_dataset_dir:
            return False
        p = Path(image_path)
        target_name = self._resolve_sample_name(image_path)
        target_stem = Path(target_name).stem
        target_suffix = Path(target_name).suffix

        try:
            shutil.copy2(str(p), os.path.join(self._sample_dataset_dir, target_name))
        except OSError as e:
            logger.warning("Failed to copy image %s: %s", p.name, e)
            return False

        ann = p.with_suffix(self._annotation_ext())
        if ann.exists():
            ann_target = f"{target_stem}{self._annotation_ext()}"
            try:
                shutil.copy2(str(ann), os.path.join(self._sample_dataset_dir, ann_target))
            except OSError as e:
                logger.warning("Failed to copy annotation %s: %s", ann.name, e)
        return True

    def _remove_from_sample_dir(self, image_path: str):
        """从加样数据集目录中删除图片及标注文件"""
        if not self._sample_dataset_dir:
            return
        norm = os.path.normcase(os.path.abspath(image_path))
        target_name = self._sample_name_map.get(norm)
        if not target_name:
            return
        target_stem = Path(target_name).stem
        ann_name = f"{target_stem}{self._annotation_ext()}"
        for name in (target_name, ann_name):
            target = os.path.join(self._sample_dataset_dir, name)
            if os.path.isfile(target):
                try:
                    os.remove(target)
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", name, e)
        self._sample_name_map.pop(norm, None)

    def _sync_annotation_to_sample_dir(self, image_path: str):
        """将标注文件同步到加样数据集目录"""
        if not self._sample_dataset_dir:
            return
        norm = os.path.normcase(os.path.abspath(image_path))
        target_name = self._sample_name_map.get(norm)
        if not target_name:
            return
        p = Path(image_path)
        ann = p.with_suffix(self._annotation_ext())
        if ann.exists():
            target_stem = Path(target_name).stem
            ann_target = f"{target_stem}{self._annotation_ext()}"
            try:
                shutil.copy2(str(ann), os.path.join(self._sample_dataset_dir, ann_target))
            except OSError as e:
                logger.warning("Failed to sync annotation %s: %s", ann.name, e)

    def _toggle_sample_mark(self):
        """切换当前图片的加样标记"""
        if not self._annotation_window:
            return
        file_path = getattr(self._annotation_window, "file_path", None)
        if not file_path:
            return
        file_path = os.path.abspath(file_path)
        self._set_sample_mark(file_path, not self.image_panel.is_sampled(file_path))

    def _on_sample_mark_from_panel(self, path: str, mark: bool):
        self._set_sample_mark(path, mark)

    def _set_sample_mark(self, image_path: str, mark: bool):
        """设置或取消指定图片的加样标记"""
        norm = os.path.normcase(os.path.abspath(image_path))

        if mark:
            target_dir = self._ensure_sample_dataset_dir()
            if not target_dir:
                return

            if hasattr(self._annotation_window, "save_file"):
                cur = getattr(self._annotation_window, "file_path", None)
                if cur and os.path.normcase(os.path.abspath(cur)) == norm:
                    self._annotation_window.save_file()

            if not self._copy_to_sample_dir(image_path):
                InfoBar.error(
                    title="加样失败",
                    content=f"无法复制 {Path(image_path).name} 到加样目录",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
                return

            self._sampled_paths.add(norm)
            self.image_panel.mark_sampled(image_path)

            InfoBar.success(
                title="已标记加样",
                content=Path(image_path).name,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=2000,
            )
        else:
            self._sampled_paths.discard(norm)
            self.image_panel.unmark_sampled(image_path)
            self._remove_from_sample_dir(image_path)

            InfoBar.info(
                title="已取消加样",
                content=Path(image_path).name,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=2000,
            )

    def _on_file_saved(self, image_path: str):
        """标注保存后，同步加样数据集中的标注文件"""
        norm = os.path.normcase(os.path.abspath(image_path))
        if norm in self._sampled_paths:
            self._sync_annotation_to_sample_dir(image_path)
