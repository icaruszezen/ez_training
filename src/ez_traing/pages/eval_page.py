"""
YOLO 模型验证页面
功能：验证配置、启动验证、日志展示、指标可视化、报告导出
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QTextCursor
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QPlainTextEdit,
    QAbstractItemView,
    QLabel,
    QDoubleSpinBox,
    QScrollArea,
    QFrame,
    QApplication,
)
from qfluentwidgets import (
    PushButton,
    PrimaryPushButton,
    TransparentPushButton,
    CardWidget,
    BodyLabel,
    SubtitleLabel,
    CaptionLabel,
    StrongBodyLabel,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    ComboBox,
    SpinBox,
    ProgressBar,
)

from ez_traing.evaluation.engine import EvaluationEngine
from ez_traing.evaluation.models import EvalConfig, EvalResult
from ez_traing.evaluation.report_generator import export_reports
from ez_traing.pages.dataset_page import ProjectManager, DatasetProject


def _get_config_dir() -> Path:
    config_dir = Path.home() / ".ez_traing"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _detect_devices() -> List[tuple]:
    devices = [("cpu", "CPU")]
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                devices.insert(0, (str(i), f"GPU {i}: {name} ({memory_gb:.1f}GB)"))
    except Exception:
        pass
    return devices


def _strip_ansi(text: str) -> str:
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class YoloEvalThread(QThread):
    """验证线程。"""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)

    def __init__(self, config: EvalConfig):
        super().__init__()
        self.config = config
        self._stop_requested = False

    def run(self):
        if self._stop_requested:
            self.finished_signal.emit(EvalResult(success=False, message="验证已取消"))
            return

        engine = EvaluationEngine()
        result = engine.run(
            self.config,
            log_callback=self.log_signal.emit,
            progress_callback=self.progress_signal.emit,
        )
        self.finished_signal.emit(result)

    def stop(self):
        self._stop_requested = True
        self.requestInterruption()
        self.log_signal.emit("[INFO] 已请求停止，当前轮次结束后退出。")


class ClickableImageLabel(QLabel):
    """可点击图片标签。"""

    clicked = pyqtSignal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_path:
            self.clicked.emit(self._image_path)
        super().mousePressEvent(event)


class EvalConfigPanel(CardWidget):
    """验证配置面板。"""

    start_eval = pyqtSignal(dict)
    stop_eval = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = ProjectManager()
        self._runs_dir = _get_config_dir() / "runs"
        self._output_root = self._runs_dir / "val"
        self._custom_model_path = ""
        self._is_running = False

        self._init_ui()
        self._refresh_datasets()
        self._refresh_train_runs()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(SubtitleLabel("验证配置", self))

        layout.addWidget(StrongBodyLabel("数据集", self))
        self.dataset_combo = ComboBox(self)
        self.dataset_combo.setMinimumWidth(220)
        layout.addWidget(self.dataset_combo)

        self.refresh_dataset_btn = TransparentPushButton("刷新数据集列表", self)
        self.refresh_dataset_btn.clicked.connect(self._refresh_datasets)
        layout.addWidget(self.refresh_dataset_btn)

        layout.addWidget(StrongBodyLabel("模型来源", self))
        self.source_combo = ComboBox(self)
        self.source_combo.addItem("训练记录", "train_runs")
        self.source_combo.addItem("本地文件", "custom")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)

        self.train_weight_box = QWidget(self)
        train_box_layout = QVBoxLayout(self.train_weight_box)
        train_box_layout.setContentsMargins(0, 0, 0, 0)
        train_box_layout.setSpacing(8)

        train_box_layout.addWidget(CaptionLabel("训练记录", self))
        self.run_list = QListWidget(self)
        self.run_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.run_list.itemSelectionChanged.connect(self._on_run_selected)
        self.run_list.setMaximumHeight(120)
        train_box_layout.addWidget(self.run_list)

        train_box_layout.addWidget(CaptionLabel("权重文件", self))
        self.weight_list = QListWidget(self)
        self.weight_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.weight_list.setMaximumHeight(100)
        train_box_layout.addWidget(self.weight_list)

        self.refresh_runs_btn = TransparentPushButton("刷新训练记录", self)
        self.refresh_runs_btn.clicked.connect(self._refresh_train_runs)
        train_box_layout.addWidget(self.refresh_runs_btn)
        layout.addWidget(self.train_weight_box)

        self.custom_weight_box = QWidget(self)
        custom_box_layout = QVBoxLayout(self.custom_weight_box)
        custom_box_layout.setContentsMargins(0, 0, 0, 0)
        custom_box_layout.setSpacing(6)

        row = QHBoxLayout()
        self.custom_model_label = CaptionLabel("未选择 .pt 文件", self)
        self.custom_model_label.setWordWrap(True)
        row.addWidget(self.custom_model_label, 1)
        self.browse_model_btn = PushButton("选择 .pt", self)
        self.browse_model_btn.setIcon(FIF.FOLDER)
        self.browse_model_btn.clicked.connect(self._choose_custom_model)
        row.addWidget(self.browse_model_btn)
        custom_box_layout.addLayout(row)
        layout.addWidget(self.custom_weight_box)
        self.custom_weight_box.setVisible(False)

        layout.addWidget(StrongBodyLabel("验证参数", self))
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(BodyLabel("Image Size:", self))
        self.imgsz_spin = SpinBox(self)
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        imgsz_layout.addWidget(self.imgsz_spin)
        layout.addLayout(imgsz_layout)

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(BodyLabel("Batch:", self))
        self.batch_spin = SpinBox(self)
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        batch_layout.addWidget(self.batch_spin)
        layout.addLayout(batch_layout)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(BodyLabel("Conf:", self))
        self.conf_spin = QDoubleSpinBox(self)
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(BodyLabel("IoU:", self))
        self.iou_spin = QDoubleSpinBox(self)
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        iou_layout.addWidget(self.iou_spin)
        layout.addLayout(iou_layout)

        device_layout = QHBoxLayout()
        device_layout.addWidget(BodyLabel("Device:", self))
        self.device_combo = ComboBox(self)
        for device_id, display in _detect_devices():
            self.device_combo.addItem(display, device_id)
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(CaptionLabel("输出目录:", self))
        self.output_label = CaptionLabel(str(self._output_root), self)
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label, 1)
        self.browse_output_btn = PushButton("浏览", self)
        self.browse_output_btn.clicked.connect(self._choose_output_dir)
        output_layout.addWidget(self.browse_output_btn)
        layout.addLayout(output_layout)

        self.model_hint_label = CaptionLabel("当前模型: 未选择", self)
        self.model_hint_label.setWordWrap(True)
        layout.addWidget(self.model_hint_label)

        reset_btn = TransparentPushButton("重置参数", self)
        reset_btn.setIcon(FIF.CANCEL_MEDIUM)
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)

        layout.addStretch()

        self.start_btn = PrimaryPushButton("开始验证", self)
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_btn)

        self.stop_btn = PushButton("停止验证", self)
        self.stop_btn.setIcon(FIF.CANCEL)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_eval.emit)
        layout.addWidget(self.stop_btn)

    def _on_source_changed(self):
        source = self.source_combo.currentData()
        self.train_weight_box.setVisible(source == "train_runs")
        self.custom_weight_box.setVisible(source == "custom")
        self._refresh_model_hint()

    def _refresh_datasets(self):
        self._project_manager = ProjectManager()
        self.dataset_combo.clear()
        for proj in self._project_manager.get_all_projects():
            self.dataset_combo.addItem(proj.name, proj.id)

    def _refresh_train_runs(self):
        self.run_list.clear()
        self.weight_list.clear()
        detect_path = self._runs_dir / "detect"
        if not detect_path.exists():
            self._refresh_model_hint()
            return
        for run_dir in sorted(detect_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            weights = run_dir / "weights"
            if not weights.exists():
                continue
            item = QListWidgetItem(run_dir.name)
            item.setData(Qt.UserRole, str(run_dir))
            self.run_list.addItem(item)
        if self.run_list.count() > 0:
            self.run_list.setCurrentRow(0)
        self._refresh_model_hint()

    def _on_run_selected(self):
        self.weight_list.clear()
        items = self.run_list.selectedItems()
        if not items:
            self._refresh_model_hint()
            return
        run_dir = Path(items[0].data(Qt.UserRole))
        weights_dir = run_dir / "weights"
        if not weights_dir.exists():
            self._refresh_model_hint()
            return
        best_item_index = -1
        for idx, weight_file in enumerate(sorted(weights_dir.glob("*.pt"))):
            item = QListWidgetItem(weight_file.name)
            item.setData(Qt.UserRole, str(weight_file))
            self.weight_list.addItem(item)
            if weight_file.name == "best.pt":
                best_item_index = idx
        if self.weight_list.count() > 0:
            self.weight_list.setCurrentRow(best_item_index if best_item_index >= 0 else 0)
        self._refresh_model_hint()

    def _choose_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择权重文件",
            str(self._runs_dir),
            "PyTorch 权重文件 (*.pt)",
        )
        if path:
            self._custom_model_path = path
            self.custom_model_label.setText(path)
            self._refresh_model_hint()

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择验证输出目录", str(self._output_root))
        if path:
            self._output_root = Path(path)
            self.output_label.setText(path)

    def _get_selected_model_path(self) -> str:
        source = self.source_combo.currentData()
        if source == "custom":
            return self._custom_model_path
        items = self.weight_list.selectedItems()
        if not items:
            return ""
        return items[0].data(Qt.UserRole)

    def _refresh_model_hint(self):
        model_path = self._get_selected_model_path()
        if not model_path:
            self.model_hint_label.setText("当前模型: 未选择")
            return
        self.model_hint_label.setText(f"当前模型: {model_path}")

    def _reset_params(self):
        self.imgsz_spin.setValue(640)
        self.batch_spin.setValue(16)
        self.conf_spin.setValue(0.25)
        self.iou_spin.setValue(0.45)
        if self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(0)

    def _on_start_clicked(self):
        if self.dataset_combo.currentIndex() < 0:
            InfoBar.warning(
                title="提示",
                content="请先选择数据集",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        project_id = self.dataset_combo.currentData()
        project = self._project_manager.get_project(project_id)
        if not project:
            InfoBar.error(
                title="错误",
                content="数据集不存在",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        model_path = self._get_selected_model_path()
        if not model_path:
            InfoBar.warning(
                title="提示",
                content="请先选择模型权重文件",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        config = {
            "project": project,
            "model_path": model_path,
            "imgsz": self.imgsz_spin.value(),
            "batch": self.batch_spin.value(),
            "device": self.device_combo.currentData(),
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "source": self.source_combo.currentData(),
            "output_root": str(self._output_root),
        }
        self.start_eval.emit(config)

    def set_running_state(self, running: bool):
        self._is_running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.dataset_combo.setEnabled(not running)
        self.source_combo.setEnabled(not running)
        self.run_list.setEnabled(not running)
        self.weight_list.setEnabled(not running)
        self.imgsz_spin.setEnabled(not running)
        self.batch_spin.setEnabled(not running)
        self.conf_spin.setEnabled(not running)
        self.iou_spin.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.refresh_dataset_btn.setEnabled(not running)
        self.refresh_runs_btn.setEnabled(not running)
        self.browse_model_btn.setEnabled(not running)
        self.browse_output_btn.setEnabled(not running)


class EvalLogPanel(CardWidget):
    """日志面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_buffer: List[str] = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(100)
        self._log_flush_timer.timeout.connect(self._flush_logs)
        self._last_progress_value = -1
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.addWidget(SubtitleLabel("验证日志", self))
        header_layout.addStretch()
        clear_btn = TransparentPushButton("清空", self)
        clear_btn.clicked.connect(self._clear_log)
        header_layout.addWidget(clear_btn)
        layout.addLayout(header_layout)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = CaptionLabel("等待验证开始...", self)
        layout.addWidget(self.progress_label)

        self.log_text = QPlainTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
            }
        """
        )
        layout.addWidget(self.log_text, 1)

    def append_log(self, text: str):
        text = _strip_ansi(text)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_buffer.append(f"[{timestamp}] {text}")
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_logs(self):
        if not self._log_buffer:
            self._log_flush_timer.stop()
            return
        self.log_text.appendPlainText("\n".join(self._log_buffer))
        self._log_buffer.clear()
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        self._log_flush_timer.stop()

    def set_progress(self, value: int):
        value = max(0, min(100, int(value)))
        if value == self._last_progress_value:
            return
        self._last_progress_value = value
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"进度: {value}%")

    def reset(self):
        self._last_progress_value = -1
        self._log_buffer.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("等待验证开始...")
        self.log_text.clear()

    def _clear_log(self):
        self._log_buffer.clear()
        self.log_text.clear()


class EvalResultPanel(CardWidget):
    """验证结果面板。"""

    export_requested = pyqtSignal()
    export_to_requested = pyqtSignal()
    copy_metrics_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._save_dir = ""
        self._chart_labels: Dict[str, QLabel] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.addWidget(SubtitleLabel("验证结果", self))
        header_layout.addStretch()
        self.open_dir_btn = PushButton("打开目录", self)
        self.open_dir_btn.setIcon(FIF.FOLDER)
        self.open_dir_btn.clicked.connect(self._open_dir)
        self.open_dir_btn.setEnabled(False)
        header_layout.addWidget(self.open_dir_btn)

        self.export_btn = PushButton("导出报告", self)
        self.export_btn.setIcon(FIF.SAVE)
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.export_btn.setEnabled(False)
        header_layout.addWidget(self.export_btn)

        self.export_to_btn = PushButton("另存报告", self)
        self.export_to_btn.setIcon(FIF.SAVE_AS)
        self.export_to_btn.clicked.connect(self.export_to_requested.emit)
        self.export_to_btn.setEnabled(False)
        header_layout.addWidget(self.export_to_btn)

        self.copy_metrics_btn = PushButton("复制指标", self)
        self.copy_metrics_btn.setIcon(FIF.COPY)
        self.copy_metrics_btn.clicked.connect(self.copy_metrics_requested.emit)
        self.copy_metrics_btn.setEnabled(False)
        header_layout.addWidget(self.copy_metrics_btn)
        layout.addLayout(header_layout)

        metrics_card = QFrame(self)
        metrics_card.setStyleSheet(
            """
            QFrame {
                background-color: #f8f9fa;
                border-radius: 8px;
            }
        """
        )
        grid = QHBoxLayout(metrics_card)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(16)

        self.metric_labels = {}
        for key in ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]:
            block = QVBoxLayout()
            title = CaptionLabel(key, self)
            value = StrongBodyLabel("0.0000", self)
            value.setAlignment(Qt.AlignCenter)
            block.addWidget(title, alignment=Qt.AlignCenter)
            block.addWidget(value, alignment=Qt.AlignCenter)
            holder = QWidget(self)
            holder.setLayout(block)
            grid.addWidget(holder)
            self.metric_labels[key] = value
        layout.addWidget(metrics_card)

        layout.addWidget(CaptionLabel("图表预览", self))
        self.chart_scroll = QScrollArea(self)
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setMinimumHeight(260)
        self.chart_container = QWidget(self)
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_layout.setContentsMargins(8, 8, 8, 8)
        self.chart_layout.setSpacing(8)
        self.chart_layout.addStretch()
        self.chart_scroll.setWidget(self.chart_container)
        layout.addWidget(self.chart_scroll, 1)

    def set_result(self, result: EvalResult):
        self._save_dir = result.save_dir or ""
        self.open_dir_btn.setEnabled(bool(self._save_dir))
        self.export_btn.setEnabled(result.success)
        self.export_to_btn.setEnabled(result.success)
        self.copy_metrics_btn.setEnabled(result.success and result.metrics is not None)

        if result.metrics:
            self.metric_labels["mAP50"].setText(f"{result.metrics.map50:.4f}")
            self.metric_labels["mAP50-95"].setText(f"{result.metrics.map50_95:.4f}")
            self.metric_labels["Precision"].setText(f"{result.metrics.precision:.4f}")
            self.metric_labels["Recall"].setText(f"{result.metrics.recall:.4f}")
            self.metric_labels["F1"].setText(f"{result.metrics.f1:.4f}")

        self._set_charts(result.artifacts)

    def _set_charts(self, artifacts: Dict[str, str]):
        while self.chart_layout.count() > 1:
            item = self.chart_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not artifacts:
            note = CaptionLabel("暂无图表输出", self)
            self.chart_layout.insertWidget(0, note)
            return

        for key, path in artifacts.items():
            if not path or not os.path.exists(path):
                continue
            title = BodyLabel(key.replace("_", " ").title(), self)
            image_label = ClickableImageLabel(path, self)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setToolTip(f"点击打开原图: {path}")
            image_label.clicked.connect(self._open_image_file)
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaledToWidth(420, Qt.SmoothTransformation)
                image_label.setPixmap(scaled)
            else:
                image_label.setText(path)
            self.chart_layout.insertWidget(self.chart_layout.count() - 1, title)
            self.chart_layout.insertWidget(self.chart_layout.count() - 1, image_label)
            self._chart_labels[key] = image_label

    def _open_dir(self):
        if self._save_dir and os.path.exists(self._save_dir):
            os.startfile(self._save_dir)

    def _open_image_file(self, image_path: str):
        if image_path and os.path.exists(image_path):
            os.startfile(image_path)

    def clear_result(self):
        self._save_dir = ""
        self.open_dir_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.export_to_btn.setEnabled(False)
        self.copy_metrics_btn.setEnabled(False)
        for key in ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]:
            self.metric_labels[key].setText("0.0000")
        self._set_charts({})

    def metrics_text(self) -> str:
        return (
            f"mAP50={self.metric_labels['mAP50'].text()}, "
            f"mAP50-95={self.metric_labels['mAP50-95'].text()}, "
            f"Precision={self.metric_labels['Precision'].text()}, "
            f"Recall={self.metric_labels['Recall'].text()}, "
            f"F1={self.metric_labels['F1'].text()}"
        )


class EvalPage(QWidget):
    """YOLO 模型验证页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._eval_thread: Optional[YoloEvalThread] = None
        self._last_result: Optional[EvalResult] = None
        self._last_config: Optional[EvalConfig] = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        header = CardWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.addWidget(SubtitleLabel("模型验证", self))
        header_layout.addStretch()
        self.status_label = CaptionLabel("就绪", self)
        header_layout.addWidget(self.status_label)
        header.setFixedHeight(40)
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal, self)
        self.config_panel = EvalConfigPanel(self)
        self.config_panel.setMinimumWidth(280)
        self.config_panel.setMaximumWidth(360)
        splitter.addWidget(self.config_panel)

        self.log_panel = EvalLogPanel(self)
        splitter.addWidget(self.log_panel)

        self.result_panel = EvalResultPanel(self)
        self.result_panel.setMinimumWidth(320)
        self.result_panel.setMaximumWidth(520)
        splitter.addWidget(self.result_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        main_layout.addWidget(splitter, 1)

    def _connect_signals(self):
        self.config_panel.start_eval.connect(self._start_eval)
        self.config_panel.stop_eval.connect(self._stop_eval)
        self.result_panel.export_requested.connect(self._export_reports)
        self.result_panel.export_to_requested.connect(self._export_reports_as)
        self.result_panel.copy_metrics_requested.connect(self._copy_metrics)

    def _start_eval(self, config_data: dict):
        if self._eval_thread and self._eval_thread.isRunning():
            return

        project: DatasetProject = config_data["project"]
        config = EvalConfig(
            dataset_name=project.name,
            dataset_dir=project.directory,
            model_path=config_data["model_path"],
            imgsz=config_data["imgsz"],
            batch=config_data["batch"],
            device=config_data["device"] or "cpu",
            conf=config_data["conf"],
            iou=config_data["iou"],
            source=config_data["source"],
            output_root=config_data["output_root"],
        )

        self._last_result = None
        self._last_config = config
        self.log_panel.reset()
        self.result_panel.clear_result()
        self.log_panel.append_log(f"[INFO] 开始验证: dataset={config.dataset_name}")

        self._eval_thread = YoloEvalThread(config)
        self._eval_thread.log_signal.connect(self.log_panel.append_log)
        self._eval_thread.progress_signal.connect(self.log_panel.set_progress)
        self._eval_thread.finished_signal.connect(self._on_eval_finished)

        self.config_panel.set_running_state(True)
        self.status_label.setText("验证中...")
        self._eval_thread.start()

    def _stop_eval(self):
        if self._eval_thread and self._eval_thread.isRunning():
            self._eval_thread.stop()
            self.status_label.setText("停止请求中...")

    def _on_eval_finished(self, result_obj: object):
        result = result_obj if isinstance(result_obj, EvalResult) else EvalResult(success=False, message="未知结果")
        self._last_result = result
        self.config_panel.set_running_state(False)

        if result.success:
            self.status_label.setText("验证完成")
            self.log_panel.append_log("[INFO] 验证成功，准备展示结果...")
            self.result_panel.set_result(result)
            self._export_reports(auto=True)
            InfoBar.success(
                title="验证完成",
                content=f"结果目录: {result.save_dir}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            self.status_label.setText("验证失败")
            self.log_panel.append_log(f"[ERROR] {result.message}")
            InfoBar.error(
                title="验证失败",
                content=result.message,
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )

        self._eval_thread = None

    def _export_reports(self, auto: bool = False):
        if not self._last_result or not self._last_result.success or not self._last_config:
            if not auto:
                InfoBar.warning(
                    title="提示",
                    content="暂无可导出的验证结果",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                )
            return

        output_dir = self._last_result.save_dir or str(_get_config_dir() / "runs" / "val")
        try:
            report_files = export_reports(self._last_result, self._last_config, output_dir)
            self.log_panel.append_log(f"[INFO] 已导出报告: {report_files.get('metrics_json', '')}")
            if not auto:
                InfoBar.success(
                    title="导出成功",
                    content=f"已导出到: {output_dir}",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=4000,
                )
        except Exception as e:
            if not auto:
                InfoBar.error(
                    title="导出失败",
                    content=str(e),
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                )

    def _export_reports_as(self):
        if not self._last_result or not self._last_result.success or not self._last_config:
            InfoBar.warning(
                title="提示",
                content="暂无可导出的验证结果",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        target_dir = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            self._last_result.save_dir or str(_get_config_dir() / "runs" / "val"),
        )
        if not target_dir:
            return

        try:
            report_files = export_reports(self._last_result, self._last_config, target_dir)
            self.log_panel.append_log(f"[INFO] 另存报告完成: {report_files.get('metrics_json', '')}")
            InfoBar.success(
                title="导出成功",
                content=f"已导出到: {target_dir}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=4000,
            )
        except Exception as e:
            InfoBar.error(
                title="导出失败",
                content=str(e),
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )

    def _copy_metrics(self):
        text = self.result_panel.metrics_text()
        QApplication.clipboard().setText(text)
        self.log_panel.append_log(f"[INFO] 已复制指标: {text}")
        InfoBar.success(
            title="复制成功",
            content="核心指标已复制到剪贴板",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2500,
        )
