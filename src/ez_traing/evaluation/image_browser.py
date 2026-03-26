"""验证结果逐张图片浏览对话框：对比标注 (GT) 与模型识别 (Pred)。"""

import gc
import logging
import queue
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QKeyEvent, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    DoubleSpinBox,
    PushButton,
    SpinBox,
)

from ez_traing.common.annotation_utils import read_yolo_boxes, read_voc_boxes
from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS
from ez_traing.ui.painting import begin_label_painter, draw_box_label

logger = logging.getLogger(__name__)

GT_COLOR_BGR: Tuple[int, int, int] = (0, 200, 0)
PRED_COLOR_BGR: Tuple[int, int, int] = (0, 0, 230)

_PREDICT_CONF = 0.01


def _draw_label_below(painter, text: str, x: int, y_top: int, bg_color_bgr):
    """Draw a text label starting below *y_top* (label grows downward)."""
    painter.save()
    fm = painter.fontMetrics()
    tw = fm.horizontalAdvance(text)
    th = fm.height()
    pad = 3
    r, g, b = bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0]
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(r, g, b))
    painter.drawRect(x, y_top, tw + 2 * pad, th + 2 * pad)
    painter.setPen(QColor(255, 255, 255))
    painter.setBrush(Qt.NoBrush)
    painter.drawText(x + pad, y_top + th + pad - fm.descent(), text)
    painter.restore()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class PredictWorker(QThread):
    """Loads a YOLO model once, then runs predict() for each queued image."""

    model_loaded = pyqtSignal()
    model_error = pyqtSignal(str)
    prediction_done = pyqtSignal(str, list)

    def __init__(self, model_path: str, iou: float, imgsz: int):
        super().__init__()
        self._model_path = model_path
        self._iou = iou
        self._imgsz = imgsz
        self._queue: queue.Queue = queue.Queue()
        self._running = True

    def run(self):
        model = None
        try:
            from ultralytics import YOLO
            model = YOLO(self._model_path)
            self.model_loaded.emit()
        except Exception as e:
            self.model_error.emit(str(e))
            return

        while self._running:
            try:
                image_path = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                results = model.predict(
                    image_path,
                    conf=_PREDICT_CONF,
                    iou=self._iou,
                    imgsz=self._imgsz,
                    verbose=False,
                )
                boxes: List[Dict] = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        label = model.names.get(cls_id, f"class_{cls_id}")
                        boxes.append({
                            "label": label,
                            "xmin": round(x1),
                            "ymin": round(y1),
                            "xmax": round(x2),
                            "ymax": round(y2),
                            "confidence": conf_val,
                        })
                self.prediction_done.emit(image_path, boxes)
            except Exception:
                self.prediction_done.emit(image_path, [])

        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def request(self, image_path: str):
        """Submit a prediction request, discarding any stale pending ones."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(image_path)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_val_images(data_yaml: str) -> Tuple[List[str], List[str]]:
    """Parse *data.yaml* and return ``(image_paths, class_names)``."""
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return [], []

    root = Path(data.get("path", ""))
    val_rel = data.get("val", "")
    val_dir = root / val_rel

    names_raw = data.get("names", {})
    if isinstance(names_raw, dict):
        class_names = [names_raw[k] for k in sorted(names_raw.keys())]
    elif isinstance(names_raw, list):
        class_names = [str(n) for n in names_raw]
    else:
        class_names = []

    image_paths: List[str] = []
    if val_dir.is_dir():
        for p in sorted(val_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                image_paths.append(str(p))

    return image_paths, class_names


def _find_gt_boxes(
    image_path: str, img_w: int, img_h: int, class_names: List[str],
) -> List[Dict]:
    """Read GT boxes — handles co-located labels and YOLO parallel dir."""
    path = Path(image_path)

    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        boxes = read_yolo_boxes(txt_path, img_w, img_h, class_names)
        if boxes:
            return boxes

    str_path = str(path)
    for sep_img, sep_lbl in [("/images/", "/labels/"), ("\\images\\", "\\labels\\")]:
        if sep_img in str_path:
            label_path = Path(
                str_path.replace(sep_img, sep_lbl, 1)
            ).with_suffix(".txt")
            if label_path.exists():
                boxes = read_yolo_boxes(label_path, img_w, img_h, class_names)
                if boxes:
                    return boxes
            break

    xml_path = path.with_suffix(".xml")
    if xml_path.exists():
        return read_voc_boxes(xml_path)

    return []


def _summarize_boxes(boxes: List[Dict], with_conf: bool = False) -> str:
    """Compact summary: ``person(2), car(1)``."""
    if not boxes:
        return "无"
    counter = Counter(b["label"] for b in boxes)
    parts: List[str] = []
    for label, count in counter.most_common():
        if with_conf:
            confs = sorted(
                (b["confidence"] for b in boxes if b["label"] == label),
                reverse=True,
            )
            conf_str = ", ".join(f"{c:.2f}" for c in confs)
            parts.append(f"{label}({count})[{conf_str}]")
        else:
            parts.append(f"{label}({count})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ImageBrowserDialog(QDialog):
    """Browse validation images with GT / Prediction overlay comparison."""

    def __init__(
        self,
        data_yaml: str,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        parent=None,
    ):
        super().__init__(parent)
        self._data_yaml = data_yaml
        self._model_path = model_path
        self._display_conf = conf
        self._iou = iou
        self._imgsz = imgsz

        self._image_paths: List[str] = []
        self._class_names: List[str] = []
        self._current_index = 0
        self._pred_cache: Dict[str, List[Dict]] = {}
        self._current_pixmap: Optional[QPixmap] = None
        self._model_ready = False
        self._worker: Optional[PredictWorker] = None

        self._image_paths, self._class_names = _scan_val_images(data_yaml)

        self._init_ui()
        self._start_worker()

        if self._image_paths:
            self._show_current()

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        self.setWindowTitle("验证结果图片浏览")
        self.setMinimumSize(900, 650)
        self.resize(1050, 750)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        nav = QWidget(self)
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.setSpacing(8)

        self.prev_btn = PushButton("< 上一张", self)
        self.prev_btn.clicked.connect(self._go_prev)
        nav_lay.addWidget(self.prev_btn)

        self.index_spin = SpinBox(self)
        self.index_spin.setRange(1, max(1, len(self._image_paths)))
        self.index_spin.setValue(1)
        self.index_spin.valueChanged.connect(self._on_index_changed)
        nav_lay.addWidget(self.index_spin)

        self.count_label = BodyLabel(f"/ {len(self._image_paths)} 张", self)
        nav_lay.addWidget(self.count_label)

        self.next_btn = PushButton("下一张 >", self)
        self.next_btn.clicked.connect(self._go_next)
        nav_lay.addWidget(self.next_btn)

        nav_lay.addStretch()

        nav_lay.addWidget(BodyLabel("置信度:", self))
        self.conf_spin = DoubleSpinBox(self)
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(self._display_conf)
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        nav_lay.addWidget(self.conf_spin)

        layout.addWidget(nav)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ddd;"
            " border-radius: 4px; }"
        )
        if not self._image_paths:
            self.image_label.setText("验证数据集中未找到图片")
        layout.addWidget(self.image_label, 1)

        info_card = CardWidget(self)
        info_lay = QVBoxLayout(info_card)
        info_lay.setContentsMargins(12, 8, 12, 8)
        info_lay.setSpacing(4)

        self.file_label = CaptionLabel("文件: -", self)
        self.file_label.setWordWrap(True)
        info_lay.addWidget(self.file_label)

        self.gt_label = BodyLabel("标注目标: -", self)
        self.gt_label.setWordWrap(True)
        info_lay.addWidget(self.gt_label)

        self.pred_label = BodyLabel("识别目标: -", self)
        self.pred_label.setWordWrap(True)
        info_lay.addWidget(self.pred_label)

        legend_lay = QHBoxLayout()
        legend_lay.setSpacing(16)
        gt_leg = QLabel("\u25a0 标注(GT)", self)
        gt_leg.setStyleSheet("color: #00C853; font-weight: bold;")
        legend_lay.addWidget(gt_leg)
        pred_leg = QLabel("\u25a0 识别(Pred)", self)
        pred_leg.setStyleSheet("color: #FF1744; font-weight: bold;")
        legend_lay.addWidget(pred_leg)
        legend_lay.addStretch()
        self.status_label = CaptionLabel("正在加载模型...", self)
        legend_lay.addWidget(self.status_label)
        info_lay.addLayout(legend_lay)

        layout.addWidget(info_card)

    # ------------------------------------------------------------ worker

    def _start_worker(self):
        if not self._model_path:
            self.status_label.setText("未指定模型")
            return
        self._worker = PredictWorker(self._model_path, self._iou, self._imgsz)
        self._worker.model_loaded.connect(self._on_model_loaded)
        self._worker.model_error.connect(self._on_model_error)
        self._worker.prediction_done.connect(self._on_prediction_done)
        self._worker.start()

    def _on_model_loaded(self):
        self._model_ready = True
        self.status_label.setText("模型已就绪")
        if self._image_paths:
            path = self._image_paths[self._current_index]
            if path not in self._pred_cache:
                self._worker.request(path)

    def _on_model_error(self, msg: str):
        self.status_label.setText(f"模型加载失败: {msg}")

    def _on_prediction_done(self, image_path: str, boxes: list):
        self._pred_cache[image_path] = boxes
        if (self._image_paths
                and self._image_paths[self._current_index] == image_path):
            self._show_current()

    # ----------------------------------------------------------- display

    def _show_current(self):
        if not self._image_paths:
            return

        image_path = self._image_paths[self._current_index]

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText(f"无法加载: {image_path}")
            self._current_pixmap = None
            return

        img_w, img_h = pixmap.width(), pixmap.height()

        gt_boxes = _find_gt_boxes(image_path, img_w, img_h, self._class_names)

        conf_threshold = self.conf_spin.value()
        pred_raw = self._pred_cache.get(image_path)
        pred_boxes: Optional[List[Dict]] = None
        if pred_raw is not None:
            pred_boxes = [
                b for b in pred_raw if b["confidence"] >= conf_threshold
            ]

        self._draw_overlays(pixmap, gt_boxes, pred_boxes)
        self._current_pixmap = pixmap
        self._update_display()

        self._update_nav()
        self.file_label.setText(f"文件: {image_path}")

        gt_summary = _summarize_boxes(gt_boxes)
        self.gt_label.setText(f"标注目标: {len(gt_boxes)}个 - {gt_summary}")

        if pred_boxes is not None:
            pred_summary = _summarize_boxes(pred_boxes, with_conf=True)
            self.pred_label.setText(
                f"识别目标: {len(pred_boxes)}个 - {pred_summary}"
            )
        else:
            self.pred_label.setText("识别目标: 加载中...")
            if self._model_ready and self._worker:
                self._worker.request(image_path)

    def _draw_overlays(
        self,
        pixmap: QPixmap,
        gt_boxes: List[Dict],
        pred_boxes: Optional[List[Dict]],
    ):
        img_w = pixmap.width()
        pen_width = max(2, min(6, img_w // 400))
        font_size = max(14, min(28, img_w // 60))

        painter = begin_label_painter(pixmap, pixel_size=font_size)

        for box in gt_boxes:
            r, g, b = GT_COLOR_BGR[2], GT_COLOR_BGR[1], GT_COLOR_BGR[0]
            pen = QPen(QColor(r, g, b))
            pen.setWidth(pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                box["xmin"], box["ymin"],
                box["xmax"] - box["xmin"],
                box["ymax"] - box["ymin"],
            )
            draw_box_label(
                painter, str(box["label"]),
                box["xmin"], box["ymin"], GT_COLOR_BGR,
            )

        if pred_boxes:
            for box in pred_boxes:
                r, g, b = PRED_COLOR_BGR[2], PRED_COLOR_BGR[1], PRED_COLOR_BGR[0]
                pen = QPen(QColor(r, g, b))
                pen.setWidth(pen_width)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(
                    box["xmin"], box["ymin"],
                    box["xmax"] - box["xmin"],
                    box["ymax"] - box["ymin"],
                )
                text = f'{box["label"]} {box["confidence"]:.2f}'
                _draw_label_below(
                    painter, text,
                    box["xmin"], box["ymax"] + 1, PRED_COLOR_BGR,
                )

        painter.end()

    def _update_display(self):
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        avail = self.image_label.size() - QSize(4, 4)
        scaled = self._current_pixmap.scaled(
            avail, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    # -------------------------------------------------------- navigation

    def _update_nav(self):
        idx = self._current_index
        total = len(self._image_paths)
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < total - 1)
        self.index_spin.blockSignals(True)
        self.index_spin.setValue(idx + 1)
        self.index_spin.blockSignals(False)

    def _go_prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self._show_current()

    def _go_next(self):
        if self._current_index < len(self._image_paths) - 1:
            self._current_index += 1
            self._show_current()

    def _on_index_changed(self, value: int):
        idx = value - 1
        if 0 <= idx < len(self._image_paths) and idx != self._current_index:
            self._current_index = idx
            self._show_current()

    def _on_conf_changed(self):
        if self._image_paths:
            self._show_current()

    # -------------------------------------------------------------- events

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Left:
            self._go_prev()
        elif key == Qt.Key_Right:
            self._go_next()
        elif key == Qt.Key_Home:
            if self._image_paths and self._current_index != 0:
                self._current_index = 0
                self._show_current()
        elif key == Qt.Key_End:
            last = len(self._image_paths) - 1
            if self._image_paths and self._current_index != last:
                self._current_index = last
                self._show_current()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        super().closeEvent(event)
