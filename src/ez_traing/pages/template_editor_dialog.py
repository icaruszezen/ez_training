"""模板编辑器对话框 - 裁剪模板区域并支持匹配测试。"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    DoubleSpinBox,
    FluentIcon as FIF,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
)

from ez_traing.template_matching.matcher import (
    MatchResult,
    PreprocessConfig,
    TemplateMatcher,
    TemplateInfo,
    imread_unicode,
)
from ez_traing.ui.painting import begin_label_painter, draw_box_label

logger = logging.getLogger(__name__)


# ======================================================================
# Crop Image Widget
# ======================================================================


class CropImageWidget(QWidget):
    """支持鼠标拖拽矩形选区、滚轮缩放、中键/右键平移的图片裁剪控件。

    内部以原图像素坐标维护选区，显示时根据缩放和平移参数映射。
    - 左键拖拽：绘制裁剪选区
    - 滚轮：以光标为中心缩放
    - 中键/右键拖拽：平移画布
    - 双击左键：适应窗口
    """

    selection_changed = pyqtSignal()
    zoom_changed = pyqtSignal(float)

    _ZOOM_MIN = 0.1
    _ZOOM_MAX = 20.0
    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_image: Optional[np.ndarray] = None
        self._display_override: Optional[np.ndarray] = None
        self._display_pixmap: Optional[QPixmap] = None

        self._crop_rect: Optional[Tuple[int, int, int, int]] = None

        # crop drag state
        self._dragging = False
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None

        # zoom & pan — _zoom is relative to "fit" scale
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # pan drag state (middle / right button)
        self._panning = False
        self._pan_anchor: Optional[QPoint] = None
        self._pan_start_x = 0.0
        self._pan_start_y = 0.0

        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        self._source_image = image.copy()
        self._display_override = None
        self._crop_rect = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_display()
        self.update()
        self.zoom_changed.emit(self._zoom)

    def set_display_override(self, image: Optional[np.ndarray]):
        """设置用于显示的替代图像（如预处理后的图像），不影响裁剪坐标。"""
        self._display_override = image
        self._update_display()
        self.update()

    def get_crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        return self._crop_rect

    def get_cropped_image(self) -> Optional[np.ndarray]:
        if self._source_image is None:
            return None
        if self._crop_rect is None:
            return self._source_image.copy()
        x, y, w, h = self._crop_rect
        return self._source_image[y : y + h, x : x + w].copy()

    def reset_selection(self):
        self._crop_rect = None
        self.update()
        self.selection_changed.emit()

    def fit_to_window(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()
        self.zoom_changed.emit(self._zoom)

    def get_zoom(self) -> float:
        return self._zoom

    # ------------------------------------------------------------------
    # Internal: display mapping
    # ------------------------------------------------------------------

    def _update_display(self):
        img = self._display_override if self._display_override is not None else self._source_image
        if img is None:
            self._display_pixmap = None
            return
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._display_pixmap = QPixmap.fromImage(qimg)

    def _fit_scale(self) -> float:
        """缩放比使图片刚好适应控件大小。"""
        if self._display_pixmap is None:
            return 1.0
        pw, ph = self._display_pixmap.width(), self._display_pixmap.height()
        ww, wh = self.width(), self.height()
        return min(ww / max(pw, 1), wh / max(ph, 1))

    def _effective_scale(self) -> float:
        return self._fit_scale() * self._zoom

    def _compute_layout(self) -> Tuple[QPoint, float]:
        if self._display_pixmap is None:
            return QPoint(0, 0), 1.0
        pw, ph = self._display_pixmap.width(), self._display_pixmap.height()
        scale = self._effective_scale()
        cx = self.width() / 2.0 + self._pan_x
        cy = self.height() / 2.0 + self._pan_y
        dx = cx - pw * scale / 2.0
        dy = cy - ph * scale / 2.0
        return QPoint(int(dx), int(dy)), scale

    def _widget_to_image(self, pos: QPoint) -> QPoint:
        offset, scale = self._compute_layout()
        if scale <= 0:
            return QPoint(0, 0)
        ix = round((pos.x() - offset.x()) / scale)
        iy = round((pos.y() - offset.y()) / scale)
        return QPoint(ix, iy)

    def _image_rect_to_widget(self, x, y, w, h) -> QRect:
        offset, scale = self._compute_layout()
        wx = int(x * scale + offset.x())
        wy = int(y * scale + offset.y())
        ww = int(w * scale)
        wh = int(h * scale)
        return QRect(wx, wy, ww, wh)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._display_pixmap is None:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图片")
            painter.end()
            return

        offset, scale = self._compute_layout()

        dest = QRect(
            offset.x(),
            offset.y(),
            int(self._display_pixmap.width() * scale),
            int(self._display_pixmap.height() * scale),
        )
        painter.drawPixmap(dest, self._display_pixmap)

        crop = self._crop_rect
        if self._dragging and self._drag_start and self._drag_current:
            p1 = self._widget_to_image(self._drag_start)
            p2 = self._widget_to_image(self._drag_current)
            crop = self._normalize_rect(p1, p2)

        if crop:
            x, y, w, h = crop
            overlay = QColor(0, 0, 0, 120)
            img_w = self._display_pixmap.width()
            img_h = self._display_pixmap.height()
            sel = self._image_rect_to_widget(x, y, w, h)
            full = self._image_rect_to_widget(0, 0, img_w, img_h)

            painter.fillRect(QRect(full.left(), full.top(), full.width(), sel.top() - full.top()), overlay)
            painter.fillRect(QRect(full.left(), sel.bottom() + 1, full.width(), full.bottom() - sel.bottom()), overlay)
            painter.fillRect(QRect(full.left(), sel.top(), sel.left() - full.left(), sel.height()), overlay)
            painter.fillRect(QRect(sel.right() + 1, sel.top(), full.right() - sel.right(), sel.height()), overlay)

            pen = QPen(QColor(0, 170, 255), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(sel)

        painter.end()

    # ------------------------------------------------------------------
    # Wheel zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        if self._display_pixmap is None:
            return

        cursor_pos = event.pos()
        img_before = self._widget_to_image(cursor_pos)

        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom = self._zoom * self._ZOOM_FACTOR
        elif delta < 0:
            new_zoom = self._zoom / self._ZOOM_FACTOR
        else:
            return

        new_zoom = max(self._ZOOM_MIN, min(new_zoom, self._ZOOM_MAX))
        self._zoom = new_zoom

        new_scale = self._effective_scale()
        pw, ph = self._display_pixmap.width(), self._display_pixmap.height()
        target_cx = cursor_pos.x() - img_before.x() * new_scale
        target_cy = cursor_pos.y() - img_before.y() * new_scale
        ideal_cx = self.width() / 2.0 - pw * new_scale / 2.0
        ideal_cy = self.height() / 2.0 - ph * new_scale / 2.0
        self._pan_x = target_cx - ideal_cx
        self._pan_y = target_cy - ideal_cy

        self.update()
        self.zoom_changed.emit(self._zoom)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if self._source_image is None:
            return

        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._pan_anchor = event.pos()
            self._pan_start_x = self._pan_x
            self._pan_start_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.pos()
            self._drag_current = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_anchor is not None:
            dx = event.pos().x() - self._pan_anchor.x()
            dy = event.pos().y() - self._pan_anchor.y()
            self._pan_x = self._pan_start_x + dx
            self._pan_y = self._pan_start_y + dy
            self.update()
            return

        if self._dragging:
            self._drag_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton) and self._panning:
            self._panning = False
            self._pan_anchor = None
            self.setCursor(Qt.CrossCursor)
            return

        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if self._drag_start and self._drag_current:
                p1 = self._widget_to_image(self._drag_start)
                p2 = self._widget_to_image(self._drag_current)
                rect = self._normalize_rect(p1, p2)
                if rect and rect[2] >= 4 and rect[3] >= 4:
                    self._crop_rect = self._clamp_rect(rect)
                else:
                    self._crop_rect = None
            self._drag_start = None
            self._drag_current = None
            self.update()
            self.selection_changed.emit()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fit_to_window()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_0:
            self.fit_to_window()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self._zoom = min(self._zoom * self._ZOOM_FACTOR, self._ZOOM_MAX)
            self.update()
            self.zoom_changed.emit(self._zoom)
        elif event.key() == Qt.Key_Minus:
            self._zoom = max(self._zoom / self._ZOOM_FACTOR, self._ZOOM_MIN)
            self.update()
            self.zoom_changed.emit(self._zoom)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_rect(
        self, p1: QPoint, p2: QPoint
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1 = min(p1.x(), p2.x()), min(p1.y(), p2.y())
        x2, y2 = max(p1.x(), p2.x()), max(p1.y(), p2.y())
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            return None
        return (x1, y1, w, h)

    def _clamp_rect(
        self, rect: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        if self._source_image is None:
            return rect
        ih, iw = self._source_image.shape[:2]
        x, y, w, h = rect
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = min(w, iw - x)
        h = min(h, ih - y)
        return (x, y, max(1, w), max(1, h))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


# ======================================================================
# ROI Draw Dialog
# ======================================================================


class RoiDrawDialog(QDialog):
    """在测试图片上绘制搜索区域 (ROI) 的对话框。"""

    def __init__(self, image: np.ndarray, current_roi: Optional[Tuple[int, int, int, int]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("绘制搜索区域 (ROI)")
        self.setMinimumSize(720, 540)
        self.resize(900, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        layout.addWidget(CaptionLabel("在图片上拖拽绘制搜索区域矩形，匹配时仅在该区域内搜索"))

        self._draw_widget = CropImageWidget()
        self._draw_widget.set_image(image)
        if current_roi:
            x, y, w, h = current_roi
            self._draw_widget._crop_rect = self._draw_widget._clamp_rect((x, y, w, h))
            self._draw_widget.update()
        layout.addWidget(self._draw_widget, 1)

        self._info_label = CaptionLabel("")
        self._draw_widget.selection_changed.connect(self._on_sel_changed)
        layout.addWidget(self._info_label)
        self._on_sel_changed()

        btn_row = QHBoxLayout()
        reset_btn = PushButton("清除选区")
        reset_btn.setIcon(FIF.SYNC)
        reset_btn.clicked.connect(self._draw_widget.reset_selection)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        ok_btn = PrimaryPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        cancel_btn = PushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _on_sel_changed(self):
        rect = self._draw_widget.get_crop_rect()
        if rect:
            x, y, w, h = rect
            self._info_label.setText(f"ROI: x={x}, y={y}, w={w}, h={h}")
        else:
            self._info_label.setText("未选择区域")

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._draw_widget.get_crop_rect()


# ======================================================================
# Test Matching Worker
# ======================================================================


class _TestMatchWorker(QThread):
    """后台执行模板匹配测试的工作线程。"""

    single_done = pyqtSignal(str, object)  # (path, MatchResult)
    all_done = pyqtSignal()

    def __init__(self, matcher: TemplateMatcher, tpl_info: TemplateInfo,
                 paths: List[str], primary_index: int, parent=None):
        super().__init__(parent)
        self._matcher = matcher
        self._tpl_info = tpl_info
        self._paths = paths
        self._primary_index = primary_index
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for i, path in enumerate(self._paths):
            if self._cancelled:
                break
            result = self._matcher.match(path, [self._tpl_info])
            self.single_done.emit(path, result)
        if not self._cancelled:
            self.all_done.emit()


# ======================================================================
# Template Editor Dialog
# ======================================================================


class TemplateEditorDialog(QDialog):
    """模板编辑器：裁剪模板区域 + 匹配测试预览。"""

    def __init__(
        self,
        image_path: str,
        default_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._image_path = image_path
        self._source_image = imread_unicode(image_path)
        self._test_image_paths: List[str] = []
        self._test_worker: Optional[_TestMatchWorker] = None

        self.setWindowTitle(f"编辑模板 - {Path(image_path).name}")
        self.setMinimumSize(1060, 720)
        self.resize(1200, 800)

        self._setup_ui(default_label or Path(image_path).stem)

        if self._source_image is not None:
            self._crop_widget.set_image(self._source_image)
            h, w = self._source_image.shape[:2]
            self._img_info_label.setText(f"原图尺寸: {w} x {h}")
        else:
            self._img_info_label.setText("无法读取图片")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self, default_label: str):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # ---- left: crop ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        left_layout.addWidget(SubtitleLabel("模板裁剪"))

        self._crop_widget = CropImageWidget()
        self._crop_widget.selection_changed.connect(self._on_selection_changed)
        left_layout.addWidget(self._crop_widget, 1)

        info_row = QHBoxLayout()
        self._img_info_label = CaptionLabel("")
        info_row.addWidget(self._img_info_label)
        info_row.addStretch()
        self._sel_info_label = CaptionLabel("选区: 无 (将使用整张图片)")
        info_row.addWidget(self._sel_info_label)
        left_layout.addLayout(info_row)

        tool_row = QHBoxLayout()
        reset_btn = PushButton("重置选区")
        reset_btn.setIcon(FIF.SYNC)
        reset_btn.clicked.connect(self._crop_widget.reset_selection)
        tool_row.addWidget(reset_btn)

        fit_btn = PushButton("适应窗口")
        fit_btn.setIcon(FIF.FIT_PAGE)
        fit_btn.clicked.connect(self._crop_widget.fit_to_window)
        tool_row.addWidget(fit_btn)

        tool_row.addStretch()
        self._zoom_label = CaptionLabel("缩放: 100%")
        self._crop_widget.zoom_changed.connect(self._on_zoom_changed)
        tool_row.addWidget(self._zoom_label)
        left_layout.addLayout(tool_row)

        left_layout.addWidget(
            CaptionLabel("滚轮缩放 | 中键/右键拖拽平移 | 双击适应窗口")
        )

        # ---- preprocessing options ----
        left_layout.addWidget(SubtitleLabel("预处理选项"))

        self._pp_grayscale_cb = CheckBox("灰度化")
        self._pp_grayscale_cb.stateChanged.connect(self._on_preprocess_changed)
        left_layout.addWidget(self._pp_grayscale_cb)

        blur_row = QHBoxLayout()
        self._pp_blur_cb = CheckBox("高斯模糊")
        self._pp_blur_cb.stateChanged.connect(self._on_preprocess_changed)
        blur_row.addWidget(self._pp_blur_cb)
        blur_row.addWidget(CaptionLabel("核大小:"))
        self._pp_blur_ksize = SpinBox()
        self._pp_blur_ksize.setRange(3, 31)
        self._pp_blur_ksize.setSingleStep(2)
        self._pp_blur_ksize.setValue(5)
        self._pp_blur_ksize.valueChanged.connect(self._force_odd_blur_ksize)
        self._pp_blur_ksize.valueChanged.connect(self._on_preprocess_changed)
        blur_row.addWidget(self._pp_blur_ksize)
        blur_row.addStretch()
        left_layout.addLayout(blur_row)

        bin_row = QHBoxLayout()
        self._pp_binary_cb = CheckBox("二值化")
        self._pp_binary_cb.stateChanged.connect(self._on_preprocess_changed)
        bin_row.addWidget(self._pp_binary_cb)
        bin_row.addWidget(CaptionLabel("阈值:"))
        self._pp_binary_thresh = SpinBox()
        self._pp_binary_thresh.setRange(0, 255)
        self._pp_binary_thresh.setValue(127)
        self._pp_binary_thresh.valueChanged.connect(self._on_preprocess_changed)
        bin_row.addWidget(self._pp_binary_thresh)
        self._pp_binary_inv_cb = CheckBox("反向")
        self._pp_binary_inv_cb.stateChanged.connect(self._on_preprocess_changed)
        bin_row.addWidget(self._pp_binary_inv_cb)
        bin_row.addStretch()
        left_layout.addLayout(bin_row)

        adapt_row = QHBoxLayout()
        self._pp_adaptive_cb = CheckBox("自适应二值化")
        self._pp_adaptive_cb.stateChanged.connect(self._on_preprocess_changed)
        adapt_row.addWidget(self._pp_adaptive_cb)
        adapt_row.addWidget(CaptionLabel("块大小:"))
        self._pp_adaptive_block = SpinBox()
        self._pp_adaptive_block.setRange(3, 99)
        self._pp_adaptive_block.setSingleStep(2)
        self._pp_adaptive_block.setValue(11)
        self._pp_adaptive_block.valueChanged.connect(self._force_odd_adaptive_block)
        self._pp_adaptive_block.valueChanged.connect(self._on_preprocess_changed)
        adapt_row.addWidget(self._pp_adaptive_block)
        adapt_row.addWidget(CaptionLabel("C:"))
        self._pp_adaptive_c = SpinBox()
        self._pp_adaptive_c.setRange(-20, 20)
        self._pp_adaptive_c.setValue(2)
        self._pp_adaptive_c.valueChanged.connect(self._on_preprocess_changed)
        adapt_row.addWidget(self._pp_adaptive_c)
        adapt_row.addStretch()
        left_layout.addLayout(adapt_row)

        canny_row = QHBoxLayout()
        self._pp_canny_cb = CheckBox("Canny 边缘检测")
        self._pp_canny_cb.stateChanged.connect(self._on_preprocess_changed)
        canny_row.addWidget(self._pp_canny_cb)
        canny_row.addWidget(CaptionLabel("低阈值:"))
        self._pp_canny_low = SpinBox()
        self._pp_canny_low.setRange(0, 500)
        self._pp_canny_low.setValue(50)
        self._pp_canny_low.valueChanged.connect(self._on_preprocess_changed)
        canny_row.addWidget(self._pp_canny_low)
        canny_row.addWidget(CaptionLabel("高阈值:"))
        self._pp_canny_high = SpinBox()
        self._pp_canny_high.setRange(0, 500)
        self._pp_canny_high.setValue(150)
        self._pp_canny_high.valueChanged.connect(self._on_preprocess_changed)
        canny_row.addWidget(self._pp_canny_high)
        canny_row.addStretch()
        left_layout.addLayout(canny_row)

        roi_row = QHBoxLayout()
        self._pp_roi_cb = CheckBox("限定搜索区域 (ROI)")
        self._pp_roi_cb.stateChanged.connect(self._on_preprocess_changed)
        roi_row.addWidget(self._pp_roi_cb)

        self._pp_roi_draw_btn = PushButton("绘制")
        self._pp_roi_draw_btn.setIcon(FIF.EDIT)
        self._pp_roi_draw_btn.clicked.connect(self._on_draw_roi)
        roi_row.addWidget(self._pp_roi_draw_btn)

        for lbl_text, attr_name in [
            ("x:", "_pp_roi_x"),
            ("y:", "_pp_roi_y"),
            ("w:", "_pp_roi_w"),
            ("h:", "_pp_roi_h"),
        ]:
            roi_row.addWidget(CaptionLabel(lbl_text))
            sb = SpinBox()
            sb.setRange(0, 99999)
            sb.setValue(0)
            sb.valueChanged.connect(self._on_preprocess_changed)
            setattr(self, attr_name, sb)
            roi_row.addWidget(sb)
        roi_row.addStretch()
        left_layout.addLayout(roi_row)

        splitter.addWidget(left)

        # ---- right: test ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        right_layout.addWidget(SubtitleLabel("匹配测试"))

        btn_row = QHBoxLayout()
        add_test_btn = PushButton("添加测试图片")
        add_test_btn.setIcon(FIF.ADD)
        add_test_btn.clicked.connect(self._on_add_test_images)
        btn_row.addWidget(add_test_btn)

        clear_test_btn = PushButton("清空")
        clear_test_btn.setIcon(FIF.DELETE)
        clear_test_btn.clicked.connect(self._on_clear_test_images)
        btn_row.addWidget(clear_test_btn)
        btn_row.addStretch()
        right_layout.addLayout(btn_row)

        self._test_list = QListWidget()
        self._test_list.setMaximumHeight(120)
        self._test_list.setIconSize(QSize(48, 48))
        self._test_list.currentRowChanged.connect(self._on_test_image_selected)
        right_layout.addWidget(self._test_list)

        param_row = QHBoxLayout()
        param_row.addWidget(StrongBodyLabel("测试阈值"))
        self._test_threshold = DoubleSpinBox()
        self._test_threshold.setRange(0.1, 1.0)
        self._test_threshold.setSingleStep(0.05)
        self._test_threshold.setValue(0.8)
        self._test_threshold.setDecimals(2)
        param_row.addWidget(self._test_threshold)
        param_row.addStretch()

        run_btn = PushButton("执行测试")
        run_btn.setIcon(FIF.PLAY)
        run_btn.clicked.connect(self._on_run_test)
        param_row.addWidget(run_btn)
        right_layout.addLayout(param_row)

        self._test_preview = QLabel()
        self._test_preview.setAlignment(Qt.AlignCenter)
        self._test_preview.setMinimumHeight(200)
        self._test_preview.setStyleSheet(
            "background: #222; border-radius: 4px;"
        )
        self._test_preview.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        right_layout.addWidget(self._test_preview, 1)

        self._test_result_label = CaptionLabel("")
        right_layout.addWidget(self._test_result_label)

        splitter.addWidget(right)
        splitter.setSizes([550, 450])

        root.addWidget(splitter, 1)

        # ---- bottom: label + buttons ----
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        bottom.addWidget(StrongBodyLabel("标签名:"))
        self._label_edit = LineEdit()
        self._label_edit.setText(default_label)
        self._label_edit.setMinimumWidth(180)
        bottom.addWidget(self._label_edit)

        bottom.addStretch()

        ok_btn = PrimaryPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        bottom.addWidget(ok_btn)

        cancel_btn = PushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(cancel_btn)

        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_cropped_image(self) -> Optional[np.ndarray]:
        return self._crop_widget.get_cropped_image()

    def get_label(self) -> str:
        text = self._label_edit.text().strip()
        return text if text else Path(self._image_path).stem

    def get_image_path(self) -> str:
        return self._image_path

    def get_preprocess_config(self) -> PreprocessConfig:
        config = PreprocessConfig()
        config.to_grayscale = self._pp_grayscale_cb.isChecked()

        if self._pp_blur_cb.isChecked():
            config.gaussian_blur_ksize = self._pp_blur_ksize.value()

        if self._pp_adaptive_cb.isChecked():
            config.use_adaptive_threshold = True
            config.adaptive_block_size = self._pp_adaptive_block.value()
            config.adaptive_c = self._pp_adaptive_c.value()
            config.binary_inverse = self._pp_binary_inv_cb.isChecked()
        elif self._pp_binary_cb.isChecked():
            config.binary_threshold = self._pp_binary_thresh.value()
            config.binary_inverse = self._pp_binary_inv_cb.isChecked()

        if self._pp_canny_cb.isChecked():
            config.canny_enabled = True
            config.canny_low = self._pp_canny_low.value()
            config.canny_high = self._pp_canny_high.value()

        if self._pp_roi_cb.isChecked():
            x = self._pp_roi_x.value()
            y = self._pp_roi_y.value()
            w = self._pp_roi_w.value()
            h = self._pp_roi_h.value()
            if w > 0 and h > 0:
                config.target_roi = (x, y, w, h)

        return config

    # ------------------------------------------------------------------
    # Selection feedback
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        rect = self._crop_widget.get_crop_rect()
        if rect:
            x, y, w, h = rect
            self._sel_info_label.setText(f"选区: {w} x {h}  (x={x}, y={y})")
        else:
            self._sel_info_label.setText("选区: 无 (将使用整张图片)")

    def _on_zoom_changed(self, zoom: float):
        self._zoom_label.setText(f"缩放: {zoom * 100:.0f}%")

    # ------------------------------------------------------------------
    # Preprocessing preview (displayed in the crop widget)
    # ------------------------------------------------------------------

    def _force_odd_blur_ksize(self, value: int):
        if value % 2 == 0:
            self._pp_blur_ksize.blockSignals(True)
            self._pp_blur_ksize.setValue(value + 1)
            self._pp_blur_ksize.blockSignals(False)

    def _force_odd_adaptive_block(self, value: int):
        if value % 2 == 0:
            self._pp_adaptive_block.blockSignals(True)
            self._pp_adaptive_block.setValue(value + 1)
            self._pp_adaptive_block.blockSignals(False)

    def _on_preprocess_changed(self):
        self._update_crop_display()

    def _update_crop_display(self):
        """根据当前预处理选项更新裁剪控件的显示图像。"""
        if self._source_image is None:
            return
        config = self.get_preprocess_config()
        has_any = (
            config.needs_grayscale
            or config.gaussian_blur_ksize > 0
            or config.canny_enabled
        )
        if has_any:
            processed = TemplateMatcher.preprocess_image(self._source_image, config)
            self._crop_widget.set_display_override(processed)
        else:
            self._crop_widget.set_display_override(None)

    # ------------------------------------------------------------------
    # ROI drawing
    # ------------------------------------------------------------------

    def _on_draw_roi(self):
        row = self._test_list.currentRow()
        if row < 0 and self._test_image_paths:
            row = 0

        test_path = ""
        if row >= 0 and self._test_image_paths:
            test_path = self._test_image_paths[row]
            img = imread_unicode(test_path)
        else:
            # 没有测试图时回退到模板原图，保证 ROI 绘制入口始终可用
            img = self._source_image.copy() if self._source_image is not None else None

        if img is None:
            self._test_result_label.setText("无法读取用于绘制 ROI 的图片")
            return

        current_roi = None
        if self._pp_roi_cb.isChecked():
            x = self._pp_roi_x.value()
            y = self._pp_roi_y.value()
            w = self._pp_roi_w.value()
            h = self._pp_roi_h.value()
            if w > 0 and h > 0:
                current_roi = (x, y, w, h)

        dlg = RoiDrawDialog(img, current_roi=current_roi, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        roi = dlg.get_roi()
        if roi:
            x, y, w, h = roi
            self._pp_roi_cb.setChecked(True)
            self._pp_roi_x.setValue(x)
            self._pp_roi_y.setValue(y)
            self._pp_roi_w.setValue(w)
            self._pp_roi_h.setValue(h)
        else:
            self._pp_roi_cb.setChecked(False)
            self._pp_roi_x.setValue(0)
            self._pp_roi_y.setValue(0)
            self._pp_roi_w.setValue(0)
            self._pp_roi_h.setValue(0)

    # ------------------------------------------------------------------
    # Test images
    # ------------------------------------------------------------------

    def _on_add_test_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择测试图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if not paths:
            return
        for p in paths:
            if p in self._test_image_paths:
                continue
            self._test_image_paths.append(p)
            pix = QPixmap(p)
            item = QListWidgetItem()
            if not pix.isNull():
                from PyQt5.QtGui import QIcon
                item.setIcon(QIcon(
                    pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                ))
            item.setText(Path(p).name)
            item.setData(Qt.UserRole, p)
            self._test_list.addItem(item)

    def _on_clear_test_images(self):
        self._test_image_paths.clear()
        self._test_list.clear()
        self._test_preview.clear()
        self._test_result_label.setText("")

    def _on_test_image_selected(self, row: int):
        if row < 0:
            return
        item = self._test_list.item(row)
        path = item.data(Qt.UserRole)
        self._show_test_image(path, boxes=[])

    # ------------------------------------------------------------------
    # Run test matching
    # ------------------------------------------------------------------

    def _on_run_test(self):
        cropped = self._crop_widget.get_cropped_image()
        if cropped is None:
            self._test_result_label.setText("没有可用的模板图像")
            return

        if not self._test_image_paths:
            self._test_result_label.setText("请先添加测试图片")
            return

        if self._test_worker and self._test_worker.isRunning():
            return

        row = self._test_list.currentRow()
        if row < 0:
            row = 0
            self._test_list.setCurrentRow(0)

        label = self.get_label()
        config = self.get_preprocess_config()

        tpl_info = TemplateMatcher.create_template_from_image(
            cropped, label, self._image_path, preprocess=config
        )
        matcher = TemplateMatcher(
            threshold=self._test_threshold.value(),
            max_candidates=50,
            multi_scale=False,
        )

        self._test_result_label.setText("正在测试...")
        self._test_primary_row = row

        self._test_worker = _TestMatchWorker(
            matcher, tpl_info, list(self._test_image_paths), row, self
        )
        self._test_worker.single_done.connect(self._on_test_single_done)
        self._test_worker.all_done.connect(self._on_test_all_done)
        self._test_worker.start()

    def _on_test_single_done(self, path: str, result: MatchResult):
        try:
            idx = self._test_image_paths.index(path)
        except ValueError:
            return

        item = self._test_list.item(idx)
        if item is None:
            return

        n = len(result.boxes)
        name = Path(path).name
        item.setText(f"{name}  ({n} 个匹配)" if n else name)

        if idx == self._test_primary_row:
            self._show_test_image(path, result.boxes)
            if result.boxes:
                self._test_result_label.setText(
                    f"找到 {len(result.boxes)} 个匹配  "
                    f"(最高置信度: {max(b.confidence for b in result.boxes):.3f})"
                )
            else:
                self._test_result_label.setText("未找到匹配")

    def _on_test_all_done(self):
        self._test_worker = None

    def _stop_test_worker(self):
        if self._test_worker and self._test_worker.isRunning():
            self._test_worker.cancel()
            self._test_worker.wait(3000)
        self._test_worker = None

    def closeEvent(self, event):
        self._stop_test_worker()
        super().closeEvent(event)

    def reject(self):
        self._stop_test_worker()
        super().reject()

    def _show_test_image(self, path: str, boxes):
        img = imread_unicode(path)
        if img is None:
            self._test_preview.setText("无法读取图片")
            return

        labels_info: List[Tuple[str, int, int, Tuple[int, int, int]]] = []
        for box in boxes:
            color = (0, 255, 0)
            cv2.rectangle(
                img,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                color,
                2,
            )
            label_text = f"{box.label} {box.confidence:.2f}"
            labels_info.append((label_text, box.x_min, box.y_min, color))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if labels_info:
            painter = begin_label_painter(pixmap)
            for text, x, y_bottom, bgr in labels_info:
                draw_box_label(painter, text, x, y_bottom, bgr)
            painter.end()

        lw = self._test_preview.width() or 400
        lh = self._test_preview.height() or 300
        scaled = pixmap.scaled(
            lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._test_preview.setPixmap(scaled)
