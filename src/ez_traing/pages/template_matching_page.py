"""模板匹配标注页面 - 使用 OpenCV 模板匹配进行数据标注"""

import json
import logging
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox as QtCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget,
    CaptionLabel,
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    ScrollArea,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
    TextEdit,
    TitleLabel,
)

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS
from ez_traing.pages.template_editor_dialog import TemplateEditorDialog
from ez_traing.prelabeling.models import BoundingBox
from ez_traing.prelabeling.voc_writer import VOCAnnotationWriter
from ez_traing.template_matching.matcher import PreprocessConfig, TemplateMatcher, TemplateInfo, imread_unicode
from ez_traing.template_matching.worker import TemplateMatchingStats, TemplateMatchingWorker
from ez_traing.ui.painting import begin_label_painter, draw_box_label
from ez_traing.ui.workers import ImageScanWorker as _ImageScanWorker

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================


_draw_box_label = draw_box_label
_begin_label_painter = begin_label_painter


# ======================================================================
# Annotable Image Label
# ======================================================================


class AnnotableImageLabel(QWidget):
    """支持缩放、平移、拖拽绘制标注框的图片预览控件。

    - 滚轮：以光标为中心缩放
    - 中键/右键拖拽：平移画布
    - 左键拖拽（标注模式）：绘制矩形框
    - 双击左键：适应窗口
    """

    box_drawn = pyqtSignal(int, int, int, int)

    _ZOOM_MIN = 0.1
    _ZOOM_MAX = 20.0
    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._annotate_mode = False
        self._pixmap: Optional[QPixmap] = None
        self._original_w = 0
        self._original_h = 0
        self._text = ""

        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        self._panning = False
        self._pan_anchor: Optional[QPoint] = None
        self._pan_start_x = 0.0
        self._pan_start_y = 0.0

        self._drawing = False
        self._draw_start: Optional[QPoint] = None
        self._draw_current: Optional[QPoint] = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # -- public API -------------------------------------------------------

    def set_annotate_mode(self, enabled: bool):
        self._annotate_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_image_with_meta(
        self,
        pixmap: QPixmap,
        original_w: int,
        original_h: int,
        reset_view: bool = True,
    ):
        self._pixmap = pixmap
        self._original_w = original_w
        self._original_h = original_h
        self._text = ""
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        if reset_view:
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
        self.update()

    def clear(self):
        self._pixmap = None
        self._text = ""
        self._original_w = 0
        self._original_h = 0
        self.update()

    def setText(self, text: str):
        self._text = text
        self._pixmap = None
        self.update()

    # -- layout computation -----------------------------------------------

    def _fit_scale(self) -> float:
        if self._pixmap is None:
            return 1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        return min(ww / max(pw, 1), wh / max(ph, 1))

    def _effective_scale(self) -> float:
        return self._fit_scale() * self._zoom

    def _compute_layout(self) -> Tuple[QPoint, float]:
        if self._pixmap is None:
            return QPoint(0, 0), 1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = self._effective_scale()
        cx = self.width() / 2.0 + self._pan_x
        cy = self.height() / 2.0 + self._pan_y
        dx = cx - pw * scale / 2.0
        dy = cy - ph * scale / 2.0
        return QPoint(int(dx), int(dy)), scale

    def _widget_to_image(self, pos: QPoint) -> Tuple[int, int]:
        offset, scale = self._compute_layout()
        if scale <= 0:
            return 0, 0
        ix = int((pos.x() - offset.x()) / scale)
        iy = int((pos.y() - offset.y()) / scale)
        ix = max(0, min(ix, self._original_w - 1))
        iy = max(0, min(iy, self._original_h - 1))
        return ix, iy

    # -- paint ------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(34, 34, 34))

        if self._pixmap is None:
            if self._text:
                painter.setPen(QColor(180, 180, 180))
                painter.drawText(self.rect(), Qt.AlignCenter, self._text)
            painter.end()
            return

        offset, scale = self._compute_layout()
        dest = QRect(
            offset.x(), offset.y(),
            int(self._pixmap.width() * scale),
            int(self._pixmap.height() * scale),
        )
        painter.drawPixmap(dest, self._pixmap)

        if self._drawing and self._draw_start and self._draw_current:
            painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(255, 165, 0, 40)))
            rect = QRect(self._draw_start, self._draw_current).normalized()
            painter.drawRect(rect)

        painter.end()

    # -- wheel zoom -------------------------------------------------------

    def wheelEvent(self, event):
        if self._pixmap is None:
            return

        cursor_pos = event.pos()
        offset_before, scale_before = self._compute_layout()
        img_x = (cursor_pos.x() - offset_before.x()) / scale_before
        img_y = (cursor_pos.y() - offset_before.y()) / scale_before

        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom = self._zoom * self._ZOOM_FACTOR
        elif delta < 0:
            new_zoom = self._zoom / self._ZOOM_FACTOR
        else:
            return

        self._zoom = max(self._ZOOM_MIN, min(new_zoom, self._ZOOM_MAX))

        new_scale = self._effective_scale()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        target_ox = cursor_pos.x() - img_x * new_scale
        target_oy = cursor_pos.y() - img_y * new_scale
        ideal_ox = self.width() / 2.0 - pw * new_scale / 2.0
        ideal_oy = self.height() / 2.0 - ph * new_scale / 2.0
        self._pan_x = target_ox - ideal_ox
        self._pan_y = target_oy - ideal_oy
        self.update()

    # -- mouse interaction ------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._pan_anchor = event.pos()
            self._pan_start_x = self._pan_x
            self._pan_start_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self._annotate_mode and event.button() == Qt.LeftButton:
            self._drawing = True
            self._draw_start = event.pos()
            self._draw_current = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_anchor is not None:
            dx = event.pos().x() - self._pan_anchor.x()
            dy = event.pos().y() - self._pan_anchor.y()
            self._pan_x = self._pan_start_x + dx
            self._pan_y = self._pan_start_y + dy
            self.update()
            return

        if self._drawing:
            self._draw_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton) and self._panning:
            self._panning = False
            self._pan_anchor = None
            self.setCursor(
                Qt.CrossCursor if self._annotate_mode else Qt.ArrowCursor
            )
            return

        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            if self._draw_start and self._draw_current:
                x1, y1 = self._widget_to_image(self._draw_start)
                x2, y2 = self._widget_to_image(self._draw_current)
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                if (x_max - x_min) > 3 and (y_max - y_min) > 3:
                    self.box_drawn.emit(x_min, y_min, x_max, y_max)
            self._draw_start = None
            self._draw_current = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self.update()


# ======================================================================
# Quick Annotate Dialog
# ======================================================================


class QuickAnnotateDialog(QDialog):
    """快速标定对话框 — 在大窗口中查看图片并绘制标注框。"""

    def __init__(
        self,
        image_path: str,
        labels: List[str],
        existing_boxes: Optional[List[BoundingBox]] = None,
        manual_indices: Optional[set] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"快速标定 - {Path(image_path).name}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1200, 800)

        self._image_path = image_path
        self._labels = labels
        self._existing_boxes = existing_boxes or []
        self._manual_indices = manual_indices or set()
        self._new_boxes: List[BoundingBox] = []

        self._cv_image = imread_unicode(image_path)
        if self._cv_image is not None:
            self._img_h, self._img_w = self._cv_image.shape[:2]
        else:
            self._img_h, self._img_w = 0, 0

        self._rgb_buf = None
        self._first_render = True
        self._setup_ui()
        self._refresh_canvas()

    # -- UI ---------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(StrongBodyLabel("标签:", self))
        self._label_combo = ComboBox(self)
        self._label_combo.addItems(self._labels)
        self._label_combo.setMinimumWidth(150)
        top.addWidget(self._label_combo)
        top.addStretch()
        self._box_count_label = CaptionLabel("已标注 0 个框", self)
        top.addWidget(self._box_count_label)
        layout.addLayout(top)

        layout.addWidget(CaptionLabel(
            "左键拖拽绘制标注框 | 滚轮缩放 | 中键/右键拖拽平移 | 双击还原", self,
        ))

        self._canvas = AnnotableImageLabel(self)
        self._canvas.setMinimumSize(640, 480)
        self._canvas.setStyleSheet("background: #1a1a1a; border-radius: 6px;")
        self._canvas.set_annotate_mode(True)
        self._canvas.box_drawn.connect(self._on_canvas_box_drawn)
        layout.addWidget(self._canvas, 1)

        bottom = QHBoxLayout()
        undo_btn = PushButton("撤销上一个框", self)
        undo_btn.setIcon(FIF.CANCEL)
        undo_btn.clicked.connect(self._on_undo)
        bottom.addWidget(undo_btn)
        bottom.addStretch()

        cancel_btn = PushButton("取消", self)
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(cancel_btn)

        ok_btn = PrimaryPushButton("确定", self)
        ok_btn.clicked.connect(self.accept)
        bottom.addWidget(ok_btn)
        layout.addLayout(bottom)

    # -- canvas rendering -------------------------------------------------

    def _refresh_canvas(self):
        if self._cv_image is None:
            self._canvas.setText("无法读取图片")
            return

        img = self._cv_image.copy()
        labels_info: List[Tuple[str, int, int, Tuple[int, int, int]]] = []

        for i, box in enumerate(self._existing_boxes):
            is_manual = i in self._manual_indices
            color = (0, 165, 255) if is_manual else (0, 255, 0)
            cv2.rectangle(
                img, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2,
            )
            label_text = (
                f"{box.label} [手动]"
                if is_manual
                else f"{box.label} {box.confidence:.2f}"
            )
            labels_info.append((label_text, box.x_min, box.y_min, color))

        for box in self._new_boxes:
            color = (255, 80, 0)
            cv2.rectangle(
                img, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2,
            )
            labels_info.append((box.label, box.x_min, box.y_min, color))

        self._rgb_buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = self._rgb_buf.shape
        qimg = QImage(self._rgb_buf.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if labels_info:
            painter = _begin_label_painter(pixmap)
            for text, x, y_bottom, bgr in labels_info:
                _draw_box_label(painter, text, x, y_bottom, bgr)
            painter.end()

        reset = self._first_render
        self._first_render = False
        self._canvas.set_image_with_meta(
            pixmap, self._img_w, self._img_h, reset_view=reset,
        )

    # -- interaction ------------------------------------------------------

    def _on_canvas_box_drawn(self, x_min, y_min, x_max, y_max):
        label = self._label_combo.currentText()
        if not label:
            return
        box = BoundingBox(
            label=label, x_min=x_min, y_min=y_min,
            x_max=x_max, y_max=y_max, confidence=1.0,
        )
        self._new_boxes.append(box)
        self._box_count_label.setText(f"已标注 {len(self._new_boxes)} 个框")
        self._refresh_canvas()

    def _on_undo(self):
        if self._new_boxes:
            self._new_boxes.pop()
            self._box_count_label.setText(
                f"已标注 {len(self._new_boxes)} 个框"
            )
            self._refresh_canvas()

    # -- public API -------------------------------------------------------

    def get_new_boxes(self) -> List[BoundingBox]:
        return list(self._new_boxes)


# ======================================================================
# Template Matching Page
# ======================================================================


class TemplateMatchingPage(QWidget):
    """模板匹配标注页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = None
        self._project_ids: List[str] = []
        self._current_project_id: Optional[str] = None
        self._image_paths: List[str] = []
        self._scan_worker: Optional[_ImageScanWorker] = None
        self._scan_cache: Dict[str, Tuple[int, List[str]]] = {}
        self._worker: Optional[TemplateMatchingWorker] = None

        # 模板列表：(path, label, TemplateInfo)
        self._template_infos: List[TemplateInfo] = []

        # 匹配结果：image_path -> list[BoundingBox]
        self._match_results: Dict[str, List[BoundingBox]] = {}
        # 勾选状态：(image_path, box_index) -> checked
        self._check_states: Dict[Tuple[str, int], bool] = {}
        # 无匹配的图片路径（匹配成功但 boxes 为空）
        self._unmatched_paths: List[str] = []
        # 手动标注框标记：(image_path, box_index)
        self._manual_box_keys: set = set()

        self._run_started_at: Optional[float] = None
        self._log_buffer: List[str] = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(100)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)

        self._voc_writer = VOCAnnotationWriter()
        self._image_cache: OrderedDict[str, "np.ndarray"] = OrderedDict()
        self._IMAGE_CACHE_MAX = 8

        self._setup_ui()

    # ==================================================================
    # UI Setup
    # ==================================================================

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = ScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(36, 20, 36, 20)
        self._content_layout.setSpacing(16)

        self._content_layout.addWidget(TitleLabel("模板匹配", self))
        self._content_layout.addSpacing(4)

        self._content_layout.addWidget(self._create_dataset_card())
        self._content_layout.addWidget(self._create_template_card())
        self._content_layout.addWidget(self._create_params_card())
        self._content_layout.addWidget(self._create_action_card())
        self._content_layout.addWidget(self._create_result_card())
        self._content_layout.addWidget(self._create_log_card())
        self._content_layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Card: 数据集选择
    # ------------------------------------------------------------------

    def _create_dataset_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("数据集选择", card))

        self.dataset_combo = ComboBox(card)
        self.dataset_combo.setPlaceholderText("请先在数据集页面创建项目")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        layout.addWidget(self.dataset_combo)

        self.dataset_info_label = CaptionLabel("", card)
        layout.addWidget(self.dataset_info_label)

        return card

    # ------------------------------------------------------------------
    # Card: 模板管理
    # ------------------------------------------------------------------

    def _create_template_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("模板图片", card))

        self._tpl_count_label = CaptionLabel("已添加 0 个模板", card)
        layout.addWidget(self._tpl_count_label)

        self._tpl_list = QListWidget(card)
        self._tpl_list.setMinimumHeight(120)
        self._tpl_list.setMaximumHeight(200)
        self._tpl_list.setSpacing(4)
        self._tpl_list.setIconSize(QSize(64, 64))
        layout.addWidget(self._tpl_list)

        btn_row = QHBoxLayout()
        add_btn = PushButton("添加模板", card)
        add_btn.setIcon(FIF.ADD)
        add_btn.clicked.connect(self._on_add_templates)
        btn_row.addWidget(add_btn)

        remove_btn = PushButton("移除选中", card)
        remove_btn.setIcon(FIF.DELETE)
        remove_btn.clicked.connect(self._on_remove_template)
        btn_row.addWidget(remove_btn)

        clear_btn = PushButton("清空全部", card)
        clear_btn.setIcon(FIF.CLOSE)
        clear_btn.clicked.connect(self._on_clear_templates)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        save_tpl_btn = PushButton("保存模板", card)
        save_tpl_btn.setIcon(FIF.SAVE)
        save_tpl_btn.clicked.connect(self._on_save_templates)
        btn_row.addWidget(save_tpl_btn)

        load_tpl_btn = PushButton("加载模板", card)
        load_tpl_btn.setIcon(FIF.FOLDER)
        load_tpl_btn.clicked.connect(self._on_load_templates)
        btn_row.addWidget(load_tpl_btn)

        layout.addLayout(btn_row)

        layout.addWidget(
            CaptionLabel(
                "每个模板的文件名（不含扩展名）将作为标注类别名称",
                card,
            )
        )

        return card

    # ------------------------------------------------------------------
    # Card: 匹配参数
    # ------------------------------------------------------------------

    def _create_params_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("匹配参数", card))

        # 阈值
        row1 = QHBoxLayout()
        row1.addWidget(StrongBodyLabel("匹配阈值", card))
        self.threshold_spin = DoubleSpinBox(card)
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.8)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("匹配得分不低于此值的候选框才会保留")
        row1.addWidget(self.threshold_spin)
        row1.addStretch()
        layout.addLayout(row1)

        # 最大候选数
        row2 = QHBoxLayout()
        row2.addWidget(StrongBodyLabel("每图最大候选数", card))
        self.max_candidates_spin = SpinBox(card)
        self.max_candidates_spin.setRange(1, 500)
        self.max_candidates_spin.setValue(50)
        row2.addWidget(self.max_candidates_spin)
        row2.addStretch()
        layout.addLayout(row2)

        # 多尺度
        self.multi_scale_cb = CheckBox("启用多尺度搜索", card)
        self.multi_scale_cb.setToolTip(
            "在多种缩放比例下搜索模板，速度较慢但可匹配不同大小的目标"
        )
        layout.addWidget(self.multi_scale_cb)

        # 跳过已有标注
        self.skip_annotated_cb = CheckBox("跳过已有标注的图片", card)
        self.skip_annotated_cb.setChecked(True)
        layout.addWidget(self.skip_annotated_cb)

        return card

    # ------------------------------------------------------------------
    # Card: 操作按钮
    # ------------------------------------------------------------------

    def _create_action_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("操作", card))

        btn_row = QHBoxLayout()

        self.start_btn = PrimaryPushButton("开始匹配", card)
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self.start_btn)

        self.cancel_btn = PushButton("取消", card)
        self.cancel_btn.setIcon(FIF.CLOSE)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.cancel_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress_bar = ProgressBar(card)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = CaptionLabel("就绪", card)
        layout.addWidget(self.progress_label)

        return card

    # ------------------------------------------------------------------
    # Card: 结果预览
    # ------------------------------------------------------------------

    def _create_result_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(SubtitleLabel("匹配结果预览", card))
        header.addStretch()

        self._select_all_btn = PushButton("全选", card)
        self._select_all_btn.clicked.connect(self._on_select_all)
        header.addWidget(self._select_all_btn)

        self._deselect_all_btn = PushButton("全不选", card)
        self._deselect_all_btn.clicked.connect(self._on_deselect_all)
        header.addWidget(self._deselect_all_btn)

        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal, card)

        # 左侧：结果图片列表
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(StrongBodyLabel("图片列表", left))

        self._result_image_list = QListWidget(left)
        self._result_image_list.currentRowChanged.connect(self._on_result_image_selected)
        left_layout.addWidget(self._result_image_list)
        splitter.addWidget(left)

        # 右侧：候选框表格 + 预览图
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(StrongBodyLabel("候选框", right))

        self._box_table = QTableWidget(right)
        self._box_table.setColumnCount(7)
        self._box_table.setHorizontalHeaderLabels(
            ["选择", "标签", "x_min", "y_min", "x_max", "y_max", "置信度"]
        )
        self._box_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._box_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._box_table.setMinimumHeight(160)
        self._box_table.setMaximumHeight(240)
        right_layout.addWidget(self._box_table)

        # 快速标定按钮行（默认隐藏，选中无匹配或有手动框的图片时显示）
        self._quick_annotate_row = QWidget(right)
        qa_layout = QHBoxLayout(self._quick_annotate_row)
        qa_layout.setContentsMargins(0, 0, 0, 0)
        qa_layout.setSpacing(8)

        self._quick_draw_btn = PushButton("快速标定", self._quick_annotate_row)
        self._quick_draw_btn.setIcon(FIF.EDIT)
        self._quick_draw_btn.clicked.connect(self._on_open_quick_annotate)
        qa_layout.addWidget(self._quick_draw_btn)

        self._undo_draw_btn = PushButton("撤销", self._quick_annotate_row)
        self._undo_draw_btn.setIcon(FIF.CANCEL)
        self._undo_draw_btn.clicked.connect(self._on_undo_draw)
        qa_layout.addWidget(self._undo_draw_btn)

        qa_layout.addStretch()
        self._quick_annotate_row.setVisible(False)
        right_layout.addWidget(self._quick_annotate_row)

        self._preview_label = AnnotableImageLabel(right)
        self._preview_label.setMinimumHeight(240)
        self._preview_label.setStyleSheet(
            "background: #222; border-radius: 4px;"
        )
        right_layout.addWidget(self._preview_label)

        splitter.addWidget(right)
        splitter.setSizes([280, 600])

        layout.addWidget(splitter)

        # 保存按钮
        save_row = QHBoxLayout()
        self.save_btn = PrimaryPushButton("保存选中结果", card)
        self.save_btn.setIcon(FIF.SAVE)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self.save_btn)

        self.save_all_btn = PrimaryPushButton("全部保存", card)
        self.save_all_btn.setIcon(FIF.SAVE)
        self.save_all_btn.setEnabled(False)
        self.save_all_btn.clicked.connect(self._on_save_all)
        save_row.addWidget(self.save_all_btn)

        save_row.addStretch()

        self._result_summary_label = CaptionLabel("", card)
        save_row.addWidget(self._result_summary_label)

        layout.addLayout(save_row)

        return card

    # ------------------------------------------------------------------
    # Card: 日志
    # ------------------------------------------------------------------

    def _create_log_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        hdr = QHBoxLayout()
        hdr.addWidget(SubtitleLabel("日志", card))
        hdr.addStretch()
        clear_btn = PushButton("清空", card)
        clear_btn.clicked.connect(self._clear_log)
        hdr.addWidget(clear_btn)
        layout.addLayout(hdr)

        self.log_text = TextEdit(card)
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

        return card

    # ==================================================================
    # Public API
    # ==================================================================

    def set_project_manager(self, manager):
        self._project_manager = manager

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_dataset_list()

    # ==================================================================
    # Dataset handling (same pattern as PrelabelingPage)
    # ==================================================================

    def _refresh_dataset_list(self):
        if not self._project_manager:
            return

        prev = self._current_project_id

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self._project_ids.clear()

        for proj in self._project_manager.get_all_projects(exclude_archived=True):
            self.dataset_combo.addItem(f"{proj.name} ({proj.image_count} 张图片)")
            self._project_ids.append(proj.id)

        if not self._project_ids:
            self.dataset_info_label.setText("请先在数据集页面创建项目")
        elif prev in self._project_ids:
            self.dataset_combo.setCurrentIndex(self._project_ids.index(prev))
        else:
            self._current_project_id = None
            self._image_paths.clear()
            self.dataset_info_label.setText("")

        self.dataset_combo.blockSignals(False)

    def _on_dataset_changed(self, index: int):
        if index < 0 or index >= len(self._project_ids):
            self._current_project_id = None
            self._image_paths.clear()
            self.dataset_info_label.setText("")
            return

        pid = self._project_ids[index]
        self._current_project_id = pid
        proj = self._project_manager.get_project(pid)
        if proj:
            self._scan_project(proj)

    def _scan_project(self, project):
        dirs = (self._project_manager.get_directories(project.id)
                if self._project_manager else [])
        if project.is_archive_root:
            if not dirs:
                self._image_paths.clear()
                self.dataset_info_label.setText("归档内没有有效目录")
                return
        elif not os.path.isdir(project.directory):
            self._image_paths.clear()
            self.dataset_info_label.setText("目录不存在")
            self._log(f"目录不存在: {project.directory}", "error")
            return

        if not project.is_archive_root:
            try:
                mtime = Path(project.directory).stat().st_mtime_ns
            except OSError:
                mtime = -1
            cached = self._scan_cache.get(project.id)
            if cached and cached[0] == mtime:
                self._image_paths = list(cached[1])
                self.dataset_info_label.setText(
                    f"已加载 {len(self._image_paths)} 张图片（缓存）"
                )
                return

        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()

        self.dataset_info_label.setText("正在扫描图片...")
        if project.is_archive_root:
            self._scan_worker = _ImageScanWorker(
                project.id, directories=dirs)
        else:
            self._scan_worker = _ImageScanWorker(
                project.id, project.directory)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_finished(self, project_id, paths, error, elapsed):
        if project_id != self._current_project_id:
            return
        if error:
            self._image_paths.clear()
            self.dataset_info_label.setText("扫描出错")
            self._log(f"扫描出错: {error}", "error")
            return
        self._image_paths = list(paths)
        self.dataset_info_label.setText(f"已加载 {len(paths)} 张图片")
        self._log(f"已加载 {len(paths)} 张图片，耗时 {elapsed:.2f}s")

        try:
            proj = self._project_manager.get_project(project_id)
            if proj:
                mtime = Path(proj.directory).stat().st_mtime_ns
                self._scan_cache[project_id] = (mtime, list(paths))
        except OSError:
            pass

    # ==================================================================
    # Template management
    # ==================================================================

    def _on_add_templates(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择模板图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if not paths:
            return

        added = 0
        for p in paths:
            dialog = TemplateEditorDialog(p, Path(p).stem, parent=self.window())
            if dialog.exec_() != QDialog.Accepted:
                continue

            cropped = dialog.get_cropped_image()
            label = dialog.get_label()
            if cropped is None:
                self._log(f"无法读取图片: {p}", "error")
                continue

            ch, cw = cropped.shape[:2]
            if any(
                t.path == p and t.label == label
                and t.width == cw and t.height == ch
                for t in self._template_infos
            ):
                self._log(f"跳过重复模板: {label} ({cw}x{ch})")
                continue

            config = dialog.get_preprocess_config()
            info = TemplateMatcher.create_template_from_image(cropped, label, p, config)
            self._template_infos.append(info)
            self._add_template_list_item(info)
            added += 1

        if added:
            self._update_tpl_count()
            self._log(f"添加了 {added} 个模板")

    def _add_template_list_item(self, info: TemplateInfo):
        if info.image is not None:
            rgb = cv2.cvtColor(info.image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
        else:
            pix = QPixmap(info.path)

        if pix.isNull():
            pix = QPixmap(64, 64)
            pix.fill(Qt.lightGray)
        else:
            pix = pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        item = QListWidgetItem()
        item.setIcon(QIcon(pix))
        item.setText(f"{info.label}  ({info.width}x{info.height})")
        item.setToolTip(info.path)
        item.setData(Qt.UserRole, info.path)
        self._tpl_list.addItem(item)

    def _on_remove_template(self):
        row = self._tpl_list.currentRow()
        if row < 0 or row >= len(self._template_infos):
            return
        self._tpl_list.takeItem(row)
        self._template_infos.pop(row)
        self._update_tpl_count()

    def _on_clear_templates(self):
        self._template_infos.clear()
        self._tpl_list.clear()
        self._update_tpl_count()

    def _update_tpl_count(self):
        self._tpl_count_label.setText(f"已添加 {len(self._template_infos)} 个模板")

    # ==================================================================
    # Template save / load
    # ==================================================================

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', "_", name)

    def _on_save_templates(self):
        if not self._template_infos:
            InfoBar.warning(
                title="提示",
                content="当前没有模板可保存",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录", ""
        )
        if not directory:
            return

        save_dir = Path(directory)
        entries: List[Dict[str, Any]] = []

        for idx, info in enumerate(self._template_infos):
            safe_label = self._sanitize_filename(info.label)
            img_filename = f"{idx}_{safe_label}.png"
            img_path = save_dir / img_filename

            if info.image is not None:
                ok, buf = cv2.imencode(".png", info.image)
                if ok:
                    with open(img_path, "wb") as f:
                        f.write(buf.tobytes())
                else:
                    self._log(f"编码模板图像失败: {info.label}", "error")
                    continue
            else:
                self._log(f"模板缺少图像数据: {info.label}", "error")
                continue

            entries.append({
                "label": info.label,
                "original_path": info.path,
                "image_file": img_filename,
                "width": info.width,
                "height": info.height,
                "preprocess": info.preprocess.to_dict(),
            })

        payload = {"version": 1, "templates": entries}
        json_path = save_dir / "templates.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"保存 templates.json 失败: {exc}", "error")
            return

        self._log(f"已保存 {len(entries)} 个模板到 {save_dir}")
        InfoBar.success(
            title="保存成功",
            content=f"已保存 {len(entries)} 个模板到所选目录",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=4000,
        )

    def _on_load_templates(self):
        json_file, _ = QFileDialog.getOpenFileName(
            self,
            "选择模板配置文件",
            "",
            "模板配置 (templates.json);;JSON 文件 (*.json);;所有文件 (*)",
        )
        if not json_file:
            return

        json_path = Path(json_file)
        base_dir = json_path.parent

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self._log(f"读取 templates.json 失败: {exc}", "error")
            InfoBar.error(
                title="加载失败",
                content=f"无法读取配置文件: {exc}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        templates_data = payload.get("templates", [])
        if not templates_data:
            InfoBar.warning(
                title="提示",
                content="配置文件中没有模板数据",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        loaded = 0
        for entry in templates_data:
            img_filename = entry.get("image_file", "")
            img_path = base_dir / img_filename
            image = imread_unicode(str(img_path))
            if image is None:
                self._log(f"无法读取模板图像: {img_path}", "error")
                continue

            label = entry.get("label", img_path.stem)
            original_path = entry.get("original_path", str(img_path))
            pp_data = entry.get("preprocess", {})
            config = PreprocessConfig.from_dict(pp_data)

            h, w = image.shape[:2]
            info = TemplateInfo(
                path=original_path,
                label=label,
                image=image,
                height=h,
                width=w,
                preprocess=config,
            )
            self._template_infos.append(info)
            self._add_template_list_item(info)
            loaded += 1

        self._update_tpl_count()
        self._log(f"从 {json_path} 加载了 {loaded} 个模板")
        InfoBar.success(
            title="加载成功",
            content=f"已加载 {loaded} 个模板",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=4000,
        )

    # ==================================================================
    # Matching flow
    # ==================================================================

    def _on_start(self):
        if not self._template_infos:
            InfoBar.warning(
                title="提示",
                content="请先添加至少一个模板图片",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        if not self._image_paths:
            InfoBar.warning(
                title="提示",
                content="当前数据集没有图片",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        matcher = TemplateMatcher(
            threshold=self.threshold_spin.value(),
            max_candidates=self.max_candidates_spin.value(),
            multi_scale=self.multi_scale_cb.isChecked(),
        )

        self._match_results.clear()
        self._check_states.clear()
        self._unmatched_paths.clear()
        self._manual_box_keys.clear()
        self._image_cache.clear()
        self._result_image_list.clear()
        self._box_table.setRowCount(0)
        self._preview_label.clear()
        self._show_quick_annotate_controls(False)
        self.save_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
        self._result_summary_label.setText("")

        self._worker = TemplateMatchingWorker(
            image_paths=self._image_paths,
            templates=self._template_infos,
            matcher=matcher,
            skip_annotated=self.skip_annotated_cb.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.image_completed.connect(self._on_image_completed)
        self._worker.finished.connect(self._on_finished)

        self._set_running(True)
        self._run_started_at = perf_counter()
        self._log("模板匹配开始")
        self._log(f"模板数: {len(self._template_infos)}, 图片数: {len(self._image_paths)}")
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._log("正在取消...", "warning")

    def _on_progress(self, current, total, message):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.progress_label.setText(f"{current}/{total} - {message}")

    def _on_image_completed(self, path, success, message, boxes):
        if boxes:
            self._match_results[path] = boxes
            for i in range(len(boxes)):
                self._check_states[(path, i)] = True

            item = QListWidgetItem(
                f"{Path(path).name}  ({len(boxes)} 个)"
            )
            item.setData(Qt.UserRole, path)
            self._result_image_list.addItem(item)
        elif success:
            self._unmatched_paths.append(path)

        if not success:
            self._log(message, "error")

    def _on_finished(self, stats: TemplateMatchingStats):
        self._set_running(False)
        self.progress_bar.setValue(100)

        elapsed = 0.0
        if self._run_started_at:
            elapsed = perf_counter() - self._run_started_at
            self._run_started_at = None

        total_boxes = sum(len(b) for b in self._match_results.values())
        summary = (
            f"完成 - 总计: {stats.total}, 匹配: {stats.matched}, "
            f"无匹配: {stats.empty}, 失败: {stats.failed}, "
            f"跳过: {stats.skipped}, 候选框: {total_boxes}"
        )
        self._log(summary)
        self._log(f"耗时: {elapsed:.2f}s")
        self.progress_label.setText(summary)

        # 将无匹配图片追加到结果列表，橙色高亮显示
        for path in self._unmatched_paths:
            item = QListWidgetItem(f"{Path(path).name}  (无匹配)")
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, "unmatched")
            item.setForeground(QBrush(QColor(255, 140, 0)))
            self._result_image_list.addItem(item)

        if total_boxes > 0 or self._unmatched_paths:
            self.save_btn.setEnabled(total_boxes > 0)
            self.save_all_btn.setEnabled(total_boxes > 0)
            parts = []
            if self._match_results:
                parts.append(
                    f"{len(self._match_results)} 张图片共 {total_boxes} 个候选框"
                )
            if self._unmatched_paths:
                parts.append(
                    f"{len(self._unmatched_paths)} 张无匹配（可快速标定）"
                )
            self._result_summary_label.setText("；".join(parts))
        else:
            self._result_summary_label.setText("未找到任何匹配")

        InfoBar.success(
            title="模板匹配完成",
            content=summary,
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=5000,
        )

        self._worker = None

    # ==================================================================
    # Result preview
    # ==================================================================

    def _on_result_image_selected(self, row):
        if row < 0:
            self._box_table.setRowCount(0)
            self._preview_label.clear()
            self._show_quick_annotate_controls(False)
            return

        item = self._result_image_list.item(row)
        path = item.data(Qt.UserRole)
        is_unmatched = item.data(Qt.UserRole + 1) == "unmatched"
        boxes = self._match_results.get(path, [])
        has_manual = any(
            (path, i) in self._manual_box_keys for i in range(len(boxes))
        )

        self._populate_box_table(path, boxes)
        self._render_preview(path, boxes)
        self._show_quick_annotate_controls(is_unmatched or has_manual)

    def _populate_box_table(self, image_path: str, boxes: List[BoundingBox]):
        self._box_table.blockSignals(True)
        self._box_table.setRowCount(len(boxes))

        for i, box in enumerate(boxes):
            cb = QtCheckBox()
            cb.setChecked(self._check_states.get((image_path, i), True))
            cb.stateChanged.connect(
                lambda state, p=image_path, idx=i: self._on_box_check_changed(
                    p, idx, state == Qt.Checked
                )
            )
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self._box_table.setCellWidget(i, 0, cb_widget)

            self._box_table.setItem(i, 1, QTableWidgetItem(box.label))
            self._box_table.setItem(i, 2, QTableWidgetItem(str(box.x_min)))
            self._box_table.setItem(i, 3, QTableWidgetItem(str(box.y_min)))
            self._box_table.setItem(i, 4, QTableWidgetItem(str(box.x_max)))
            self._box_table.setItem(i, 5, QTableWidgetItem(str(box.y_max)))
            self._box_table.setItem(
                i, 6, QTableWidgetItem(f"{box.confidence:.3f}")
            )

            for col in range(1, 7):
                it = self._box_table.item(i, col)
                if it:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)

        self._box_table.blockSignals(False)

    def _on_box_check_changed(self, image_path: str, idx: int, checked: bool):
        self._check_states[(image_path, idx)] = checked
        boxes = self._match_results.get(image_path, [])
        if boxes:
            self._render_preview(image_path, boxes)

    def _read_image_cached(self, image_path: str) -> Optional[np.ndarray]:
        if image_path in self._image_cache:
            self._image_cache.move_to_end(image_path)
            return self._image_cache[image_path]
        img = imread_unicode(image_path)
        if img is None:
            return None
        self._image_cache[image_path] = img
        while len(self._image_cache) > self._IMAGE_CACHE_MAX:
            self._image_cache.popitem(last=False)
        return img

    def _render_preview(self, image_path: str, boxes: List[BoundingBox]):
        """在预览区绘制带候选框的图片。"""
        cached = self._read_image_cached(image_path)
        if cached is None:
            self._preview_label.setText("无法读取图片")
            return
        img = cached.copy()

        h_orig, w_orig = img.shape[:2]
        labels_info: List[Tuple[str, int, int, Tuple[int, int, int]]] = []

        for i, box in enumerate(boxes):
            checked = self._check_states.get((image_path, i), True)
            is_manual = (image_path, i) in self._manual_box_keys
            if is_manual:
                color = (255, 165, 0)
            elif checked:
                color = (0, 255, 0)
            else:
                color = (128, 128, 128)
            cv2.rectangle(img, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2)
            label_text = (
                f"{box.label} [手动]"
                if is_manual
                else f"{box.label} {box.confidence:.2f}"
            )
            labels_info.append((label_text, box.x_min, box.y_min, color))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if labels_info:
            painter = _begin_label_painter(pixmap)
            for text, x, y_bottom, bgr in labels_info:
                _draw_box_label(painter, text, x, y_bottom, bgr)
            painter.end()

        self._preview_label.set_image_with_meta(pixmap, w_orig, h_orig)

    # ==================================================================
    # Quick annotation (快速标定)
    # ==================================================================

    def _show_quick_annotate_controls(self, visible: bool):
        self._quick_annotate_row.setVisible(visible)

    def _on_open_quick_annotate(self):
        row = self._result_image_list.currentRow()
        if row < 0:
            return
        item = self._result_image_list.item(row)
        path = item.data(Qt.UserRole)

        labels = list(dict.fromkeys(t.label for t in self._template_infos))
        if not labels:
            return

        existing = self._match_results.get(path, [])
        manual_idx = {
            i for i in range(len(existing))
            if (path, i) in self._manual_box_keys
        }

        dialog = QuickAnnotateDialog(
            path, labels, existing, manual_idx, parent=self.window(),
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        new_boxes = dialog.get_new_boxes()
        if not new_boxes:
            return

        if path not in self._match_results:
            self._match_results[path] = []
        boxes = self._match_results[path]
        for box in new_boxes:
            idx = len(boxes)
            boxes.append(box)
            self._check_states[(path, idx)] = True
            self._manual_box_keys.add((path, idx))

        if path in self._unmatched_paths:
            self._unmatched_paths.remove(path)
        item.setText(f"{Path(path).name}  ({len(boxes)} 个)")
        item.setData(Qt.UserRole + 1, None)
        item.setForeground(QBrush())

        self._populate_box_table(path, boxes)
        self._render_preview(path, boxes)

        total_boxes = sum(len(b) for b in self._match_results.values())
        if total_boxes > 0:
            self.save_btn.setEnabled(True)
            self.save_all_btn.setEnabled(True)

        self._log(
            f"快速标定: {Path(path).name} 添加了 {len(new_boxes)} 个标注框"
        )

    def _on_undo_draw(self):
        row = self._result_image_list.currentRow()
        if row < 0:
            return
        item = self._result_image_list.item(row)
        path = item.data(Qt.UserRole)
        boxes = self._match_results.get(path, [])
        if not boxes:
            return

        last_idx = len(boxes) - 1
        if (path, last_idx) not in self._manual_box_keys:
            return

        boxes.pop()
        self._check_states.pop((path, last_idx), None)
        self._manual_box_keys.discard((path, last_idx))

        if not boxes:
            del self._match_results[path]
            self._unmatched_paths.append(path)
            item.setText(f"{Path(path).name}  (无匹配)")
            item.setData(Qt.UserRole + 1, "unmatched")
            item.setForeground(QBrush(QColor(255, 140, 0)))
        else:
            item.setText(f"{Path(path).name}  ({len(boxes)} 个)")

        self._populate_box_table(path, boxes)
        self._render_preview(path, boxes)

    # ==================================================================
    # Select / Deselect all
    # ==================================================================

    def _on_select_all(self):
        for key in self._check_states:
            self._check_states[key] = True
        self._refresh_current_table()

    def _on_deselect_all(self):
        for key in self._check_states:
            self._check_states[key] = False
        self._refresh_current_table()

    def _refresh_current_table(self):
        row = self._result_image_list.currentRow()
        if row >= 0:
            item = self._result_image_list.item(row)
            path = item.data(Qt.UserRole)
            boxes = self._match_results.get(path, [])
            self._populate_box_table(path, boxes)
            self._render_preview(path, boxes)

    # ==================================================================
    # Save
    # ==================================================================

    def _on_save(self):
        self._do_save(only_checked=True)

    def _on_save_all(self):
        self._do_save(only_checked=False)

    def _do_save(self, only_checked: bool):
        saved_count = 0
        merged_count = 0
        total_boxes = 0

        for image_path, boxes in self._match_results.items():
            if only_checked:
                selected = [
                    box for i, box in enumerate(boxes)
                    if self._check_states.get((image_path, i), True)
                ]
            else:
                selected = list(boxes)

            if not selected:
                continue

            xml_path = str(Path(image_path).with_suffix(".xml"))

            try:
                image_size = self._voc_writer.get_image_size(image_path)
                has_existing = Path(xml_path).exists()

                if has_existing:
                    self._voc_writer.save_merged_annotation(
                        image_path, image_size, selected, output_path=xml_path,
                    )
                    merged_count += 1
                else:
                    self._voc_writer.save_annotation(
                        image_path, image_size, selected, output_path=xml_path,
                    )

                saved_count += 1
                total_boxes += len(selected)
            except Exception as exc:
                self._log(f"保存失败 {Path(image_path).name}: {exc}", "error")

        title = "全部保存完成" if not only_checked else "保存完成"
        summary = f"已保存 {saved_count} 张图片的 {total_boxes} 个标注框"
        if merged_count:
            summary += f"（其中 {merged_count} 张与已有标注合并）"

        self._log(summary)
        InfoBar.success(
            title=title,
            content=summary,
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=5000,
        )

    # ==================================================================
    # UI state
    # ==================================================================

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.cancel_btn.setVisible(running)
        self.dataset_combo.setEnabled(not running)
        self.threshold_spin.setEnabled(not running)
        self.max_candidates_spin.setEnabled(not running)
        self.multi_scale_cb.setEnabled(not running)
        self.skip_annotated_cb.setEnabled(not running)
        has_results = not running and bool(self._match_results)
        self.save_btn.setEnabled(has_results)
        self.save_all_btn.setEnabled(has_results)

    # ==================================================================
    # Logging
    # ==================================================================

    def _log(self, message: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = level.upper()
        self._log_buffer.append(f"[{ts}] [{tag}] {message}")
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_log_buffer(self):
        if not self._log_buffer:
            self._log_flush_timer.stop()
            return
        chunk = "\n".join(self._log_buffer)
        self._log_buffer.clear()
        self.log_text.append(chunk)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        self._log_flush_timer.stop()

    def _clear_log(self):
        self._log_buffer.clear()
        self.log_text.clear()
