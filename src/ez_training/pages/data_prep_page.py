"""训练前数据准备页面。"""

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CaptionLabel,
    CheckBox,
    ComboBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
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

from ez_training.common.constants import get_config_dir, open_path
from ez_training.data_prep.augmentation import get_augmentation_specs, is_albumentations_available
from ez_training.data_prep.models import DataPrepConfig, DataPrepSummary
from ez_training.data_prep.pipeline import DataPrepPipeline


def _get_default_output_dir() -> str:
    output_dir = get_config_dir() / "prepared_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


class DataPrepWorker(QThread):
    """数据准备后台线程。"""

    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str, object)

    def __init__(self, config: DataPrepConfig):
        super().__init__()
        self._config = config
        self._cancelled = False

    def run(self):
        try:
            pipeline = DataPrepPipeline(self._config)
            summary = pipeline.run(
                log_callback=self.log.emit,
                progress_callback=self.progress.emit,
                is_cancelled=lambda: self._cancelled,
            )
            if self._cancelled:
                self.finished.emit(False, "任务已取消", None)
                return
            self.finished.emit(True, "数据准备完成", summary)
        except Exception as e:
            self.finished.emit(False, str(e), None)

    def cancel(self):
        self._cancelled = True
        self.log.emit("收到取消请求，正在停止...")


class DataPrepPage(QWidget):
    """训练前数据准备页面。"""

    _STATE_FILE = "data_prep_ui_state.json"
    _DEFAULT_AUG_METHODS = {"hflip", "brightness_contrast", "gauss_noise"}

    _AUGMENTATION_HELP_TEXTS: Dict[str, str] = {
        "hflip": "作用：左右翻转，提升模型对目标左右朝向变化的鲁棒性。\n建议场景：目标本身左右对称或方向不敏感（如通用工业件、自然物体）。",
        "vflip": "作用：上下翻转，扩展垂直方向姿态分布。\n建议场景：拍摄方向可能颠倒或上下方向不重要时使用；若任务对上下方向敏感请谨慎开启。",
        "rotate": "作用：随机小角度旋转，增强模型对相机倾斜和目标轻微转动的适应能力。\n建议场景：手持拍摄、安装角度不稳定或目标有旋转变化。",
        "shift_scale_rotate": "作用：联合平移/缩放/旋转，模拟真实拍摄中的构图和尺度扰动。\n建议场景：目标在画面中的位置和大小变化较大时。",
        "affine": "作用：仿射几何变换（缩放、平移等），提高几何扰动下的泛化。\n建议场景：镜头角度变化不大，但存在轻微形变或视角偏移。",
        "perspective": "作用：透视变换，模拟视角变化导致的近大远小和梯形畸变。\n建议场景：斜拍、广角或相机位置变化明显的场景。",
        "random_resized_crop": "作用：随机裁剪并缩放，强化局部特征学习并提升不同目标尺度的适配能力。\n建议场景：目标可能只占图像局部区域，或需要提升小目标/局部目标识别能力。",
        "brightness_contrast": "作用：调整亮度与对比度，增强对光照变化和曝光差异的鲁棒性。\n建议场景：白天/夜晚、阴影、逆光、不同曝光条件并存的数据。",
        "hsv": "作用：扰动色相、饱和度与明度，降低模型对颜色分布偏移的敏感性。\n建议场景：不同相机、白平衡或环境光导致颜色变化明显时。",
        "rgb_shift": "作用：分别偏移 RGB 通道，模拟传感器与色彩响应差异。\n建议场景：多设备采集、跨相机部署或颜色漂移问题较突出时。",
        "clahe": "作用：局部对比度增强，突出暗部细节和纹理。\n建议场景：低照度、雾感、对比度偏低且细节不清的图像。",
        "gamma": "作用：Gamma 变换，模拟不同亮度响应曲线。\n建议场景：图像整体偏暗/偏亮，或存在不同相机成像风格时。",
        "gaussian_blur": "作用：高斯模糊，模拟轻微失焦与成像平滑。\n建议场景：拍摄可能轻微虚焦、细节偶有模糊。",
        "motion_blur": "作用：运动模糊，模拟拍摄或目标移动造成的拖影。\n建议场景：运动场景、快门较慢或设备抖动导致模糊。",
        "gauss_noise": "作用：添加高斯噪声，提升对传感器噪声和压缩噪声的鲁棒性。\n建议场景：弱光高 ISO、设备噪声较大或图像压缩较重。",
        "median_blur": "作用：中值模糊，可抑制椒盐噪声并保留部分边缘结构。\n建议场景：存在离散噪点、脉冲噪声或压缩伪影。",
        "coarse_dropout": "作用：随机遮挡图像局部区域，提升被遮挡情况下的识别能力。\n建议场景：目标经常被遮挡、重叠或视野不完整。",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = None
        self._project_ids: List[str] = []
        self._current_project_id: Optional[str] = None
        self._worker: Optional[DataPrepWorker] = None
        self._method_checkboxes: Dict[str, CheckBox] = {}
        self._run_started_at: Optional[float] = None
        self._last_progress_percent = -1
        self._last_progress_update_at = 0.0
        self._log_count = 0
        self._log_buffer: List[str] = []
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(100)
        self._log_timer.timeout.connect(self._flush_log_buffer)
        self._ui_state_path = get_config_dir() / self._STATE_FILE
        self._restoring_ui_state = False
        self._save_state_timer = QTimer(self)
        self._save_state_timer.setSingleShot(True)
        self._save_state_timer.setInterval(300)
        self._save_state_timer.timeout.connect(self._do_save_ui_state)
        self._setup_ui()
        self._bind_persistence_signals()
        self._load_ui_state()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget(self)
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(36, 20, 36, 20)
        self._content_layout.setSpacing(16)

        self._content_layout.addWidget(TitleLabel("训练前数据准备", self))
        self._content_layout.addWidget(self._create_dataset_card())
        self._content_layout.addWidget(self._create_split_card())
        self._content_layout.addWidget(self._create_augmentation_card())
        self._content_layout.addWidget(self._create_output_card())
        self._content_layout.addWidget(self._create_action_card())
        self._content_layout.addWidget(self._create_log_card())
        self._content_layout.addStretch()

        scroll_area.setWidget(content)
        main_layout.addWidget(scroll_area)

    def _create_dataset_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("数据源", card))
        layout.addWidget(StrongBodyLabel("选择数据集项目", card))
        self.dataset_combo = ComboBox(card)
        self.dataset_combo.setPlaceholderText("请先在数据集页面创建项目")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        layout.addWidget(self.dataset_combo)

        btn_layout = QHBoxLayout()
        self.refresh_dataset_btn = PushButton("刷新项目列表", card)
        self.refresh_dataset_btn.setIcon(FIF.SYNC)
        self.refresh_dataset_btn.clicked.connect(self._refresh_dataset_list)
        btn_layout.addWidget(self.refresh_dataset_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.dataset_info_label = CaptionLabel("未选择数据集", card)
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)
        return card

    def _create_split_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("训练集 / 验证集划分", card))

        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(BodyLabel("训练集比例 (%)", card))
        self.train_ratio_spin = SpinBox(card)
        self.train_ratio_spin.setRange(50, 95)
        self.train_ratio_spin.setValue(80)
        self.train_ratio_spin.valueChanged.connect(self._update_ratio_hint)
        ratio_layout.addWidget(self.train_ratio_spin)
        ratio_layout.addStretch()
        layout.addLayout(ratio_layout)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(BodyLabel("随机种子", card))
        self.seed_spin = SpinBox(card)
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addStretch()
        layout.addLayout(seed_layout)

        self.ratio_hint_label = CaptionLabel("", card)
        layout.addWidget(self.ratio_hint_label)
        layout.addWidget(
            CaptionLabel("已启用防泄露划分：同源命名样本会按组分到同一集合", card)
        )
        self._update_ratio_hint()
        return card

    def _create_augmentation_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("数据增强", card))
        self.enable_aug_cb = CheckBox("启用数据增强", card)
        self.enable_aug_cb.setChecked(True)
        self.enable_aug_cb.toggled.connect(self._on_aug_toggled)
        layout.addWidget(self.enable_aug_cb)

        count_layout = QHBoxLayout()
        count_layout.addWidget(BodyLabel("每张图生成增强样本数", card))
        self.aug_count_spin = SpinBox(card)
        self.aug_count_spin.setRange(1, 10)
        self.aug_count_spin.setValue(1)
        count_layout.addWidget(self.aug_count_spin)
        count_layout.addStretch()
        layout.addLayout(count_layout)

        scope_layout = QHBoxLayout()
        scope_layout.addWidget(BodyLabel("增强作用范围", card))
        self.aug_scope_combo = ComboBox(card)
        self.aug_scope_combo.addItem("仅训练集", userData="train")
        self.aug_scope_combo.addItem("训练集 + 验证集", userData="both")
        scope_layout.addWidget(self.aug_scope_combo)
        scope_layout.addStretch()
        layout.addLayout(scope_layout)

        layout.addWidget(CaptionLabel("可多选增强方法（越多越强，但耗时更长）", card))
        layout.addWidget(CaptionLabel("点击每个方法后的 ? 可查看作用和适用场景", card))

        self.aug_methods_container = QWidget(card)
        methods_layout = QGridLayout(self.aug_methods_container)
        methods_layout.setContentsMargins(0, 0, 0, 0)
        methods_layout.setHorizontalSpacing(18)
        methods_layout.setVerticalSpacing(6)

        specs = get_augmentation_specs()
        for i, (key, display_name) in enumerate(specs):
            method_row = QWidget(self.aug_methods_container)
            method_row_layout = QHBoxLayout(method_row)
            method_row_layout.setContentsMargins(0, 0, 0, 0)
            method_row_layout.setSpacing(4)

            cb = CheckBox(display_name, method_row)
            cb.setChecked(key in self._DEFAULT_AUG_METHODS)
            method_row_layout.addWidget(cb)

            help_btn = PushButton("?", method_row)
            help_btn.setFixedSize(24, 24)
            help_btn.setToolTip(self._AUGMENTATION_HELP_TEXTS.get(key, "暂无说明"))
            help_btn.clicked.connect(
                lambda _, k=key, n=display_name: self._show_aug_method_help(k, n)
            )
            method_row_layout.addWidget(help_btn)
            method_row_layout.addStretch()

            row = i // 3
            col = i % 3
            methods_layout.addWidget(method_row, row, col)
            self._method_checkboxes[key] = cb
        layout.addWidget(self.aug_methods_container)

        self.aug_hint_label = CaptionLabel("当前已选择 3 种增强方法", card)
        layout.addWidget(self.aug_hint_label)

        for cb in self._method_checkboxes.values():
            cb.toggled.connect(self._update_aug_hint)

        self._update_aug_hint()
        self._on_aug_toggled(self.enable_aug_cb.isChecked())
        return card

    def _create_output_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("输出设置", card))
        layout.addWidget(StrongBodyLabel("输出目录", card))

        row = QHBoxLayout()
        self.output_dir_edit = LineEdit(card)
        self.output_dir_edit.setText(_get_default_output_dir())
        row.addWidget(self.output_dir_edit, 1)
        self.browse_output_btn = PushButton("浏览", card)
        self.browse_output_btn.setIcon(FIF.FOLDER)
        self.browse_output_btn.clicked.connect(self._browse_output_dir)
        row.addWidget(self.browse_output_btn)
        layout.addLayout(row)

        self.skip_unlabeled_cb = CheckBox("仅处理有 VOC 标注的图片（跳过无标注）", card)
        self.skip_unlabeled_cb.setChecked(True)
        layout.addWidget(self.skip_unlabeled_cb)

        self.overwrite_cb = CheckBox("覆盖输出目录中的旧结果（images/labels/data.yaml/classes.txt）", card)
        self.overwrite_cb.setChecked(True)
        layout.addWidget(self.overwrite_cb)

        self.custom_classes_cb = CheckBox("使用自定义类别文件（classes.txt）", card)
        self.custom_classes_cb.setChecked(False)
        self.custom_classes_cb.toggled.connect(self._on_custom_classes_toggled)
        layout.addWidget(self.custom_classes_cb)

        classes_row = QHBoxLayout()
        self.custom_classes_edit = LineEdit(card)
        self.custom_classes_edit.setPlaceholderText("选择 classes.txt 文件路径")
        self.custom_classes_edit.setEnabled(False)
        classes_row.addWidget(self.custom_classes_edit, 1)
        self.browse_classes_btn = PushButton("浏览", card)
        self.browse_classes_btn.setIcon(FIF.DOCUMENT)
        self.browse_classes_btn.setEnabled(False)
        self.browse_classes_btn.clicked.connect(self._browse_custom_classes)
        classes_row.addWidget(self.browse_classes_btn)
        layout.addLayout(classes_row)

        self.custom_classes_hint = CaptionLabel(
            "文件每行一个类别名称，类别顺序将决定 YOLO 标注中的 class_id", card
        )
        self.custom_classes_hint.setEnabled(False)
        layout.addWidget(self.custom_classes_hint)

        return card

    def _create_action_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("执行", card))
        btn_layout = QHBoxLayout()
        self.start_btn = PrimaryPushButton("开始准备数据", card)
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = PushButton("取消", card)
        self.cancel_btn.setIcon(FIF.CLOSE)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self.cancel_btn)

        self.open_output_btn = PushButton("打开输出目录", card)
        self.open_output_btn.setIcon(FIF.FOLDER)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        btn_layout.addWidget(self.open_output_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.progress_bar = ProgressBar(card)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = CaptionLabel("就绪", card)
        layout.addWidget(self.progress_label)
        return card

    def _create_log_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(SubtitleLabel("日志", card))
        header.addStretch()
        clear_btn = PushButton("清空", card)
        clear_btn.clicked.connect(self._clear_log)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self.log_edit = TextEdit(card)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(220)
        self.log_edit.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_edit)
        return card

    def set_project_manager(self, manager) -> None:
        self._project_manager = manager

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_dataset_list()

    def _refresh_dataset_list(self):
        if self._project_manager is None:
            return

        prev_id = self._current_project_id
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self._project_ids.clear()

        projects = self._project_manager.get_all_projects(exclude_archived=True)
        for proj in projects:
            self.dataset_combo.addItem(f"{proj.name} ({proj.image_count} 张)")
            self._project_ids.append(proj.id)

        if not projects:
            self.dataset_info_label.setText("请先在“数据集”页面创建项目")
            self._current_project_id = None
        elif prev_id in self._project_ids:
            idx = self._project_ids.index(prev_id)
            self.dataset_combo.setCurrentIndex(idx)
            self._current_project_id = prev_id
            self._update_dataset_info(prev_id)
        else:
            self.dataset_combo.setCurrentIndex(0)
            self._current_project_id = self._project_ids[0]
            self._update_dataset_info(self._current_project_id)

        self.dataset_combo.blockSignals(False)
        self._save_ui_state()

    def _on_dataset_changed(self, index: int):
        if index < 0 or index >= len(self._project_ids):
            self._current_project_id = None
            self.dataset_info_label.setText("未选择数据集")
            return
        self._current_project_id = self._project_ids[index]
        self._update_dataset_info(self._current_project_id)
        self._save_ui_state()

    def _update_dataset_info(self, project_id: str):
        if self._project_manager is None:
            return
        project = self._project_manager.get_project(project_id)
        if not project:
            return
        if project.is_archive_root:
            dirs = self._project_manager.get_archive_directories(project.archive_id)
            self.dataset_info_label.setText(
                f"归档: {project.name}\n包含 {len(dirs)} 个目录\n记录图片数: {project.image_count}"
            )
        else:
            self.dataset_info_label.setText(
                f"项目: {project.name}\n目录: {project.directory}\n记录图片数: {project.image_count}"
            )

    def _update_ratio_hint(self):
        train = self.train_ratio_spin.value()
        val = 100 - train
        self.ratio_hint_label.setText(f"当前划分比例: train {train}% / val {val}%")

    def _on_aug_toggled(self, enabled: bool):
        self.aug_count_spin.setEnabled(enabled)
        self.aug_scope_combo.setEnabled(enabled)
        self.aug_methods_container.setEnabled(enabled)
        self.aug_hint_label.setEnabled(enabled)

    def _update_aug_hint(self):
        selected = len(self._selected_methods())
        self.aug_hint_label.setText(f"当前已选择 {selected} 种增强方法")

    def _show_aug_method_help(self, method_key: str, display_name: str):
        detail = self._AUGMENTATION_HELP_TEXTS.get(
            method_key,
            "作用：用于提升模型对该类变化的鲁棒性。\n建议场景：当真实数据中存在对应扰动时可启用。",
        )
        InfoBar.info(
            title=display_name,
            content=detail,
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=8000,
        )

    def _selected_methods(self) -> List[str]:
        return [key for key, cb in self._method_checkboxes.items() if cb.isChecked()]

    def _bind_persistence_signals(self) -> None:
        self.train_ratio_spin.valueChanged.connect(self._save_ui_state)
        self.seed_spin.valueChanged.connect(self._save_ui_state)
        self.enable_aug_cb.toggled.connect(self._save_ui_state)
        self.aug_count_spin.valueChanged.connect(self._save_ui_state)
        self.aug_scope_combo.currentIndexChanged.connect(self._save_ui_state)
        self.output_dir_edit.textChanged.connect(self._save_ui_state)
        self.skip_unlabeled_cb.toggled.connect(self._save_ui_state)
        self.overwrite_cb.toggled.connect(self._save_ui_state)
        for cb in self._method_checkboxes.values():
            cb.toggled.connect(self._save_ui_state)
        self.custom_classes_cb.toggled.connect(self._save_ui_state)
        self.custom_classes_edit.textChanged.connect(self._save_ui_state)

    def _ui_state_dir(self) -> Path:
        state_dir = self._ui_state_path.parent
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def _read_ui_state(self) -> Dict[str, Any]:
        if not self._ui_state_path.exists():
            return {}
        try:
            with open(self._ui_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            return {}
        return {}

    @staticmethod
    def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _safe_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    def _get_aug_scope(self) -> str:
        scope = self.aug_scope_combo.currentData()
        if scope in {"train", "both"}:
            return scope
        scope_text = self.aug_scope_combo.currentText()
        return "both" if "验证" in scope_text else "train"

    def _set_aug_scope(self, scope: str) -> None:
        for idx in range(self.aug_scope_combo.count()):
            if self.aug_scope_combo.itemData(idx) == scope:
                self.aug_scope_combo.setCurrentIndex(idx)
                return
        self.aug_scope_combo.setCurrentIndex(0)

    def _load_ui_state(self) -> None:
        state = self._read_ui_state()
        if not state:
            return

        self._restoring_ui_state = True
        try:
            project_id = state.get("project_id")
            if isinstance(project_id, str) and project_id.strip():
                self._current_project_id = project_id.strip()

            self.train_ratio_spin.setValue(
                self._safe_int(
                    state.get("train_ratio"), default=80, minimum=50, maximum=95
                )
            )
            self.seed_spin.setValue(
                self._safe_int(
                    state.get("random_seed"), default=42, minimum=0, maximum=999999
                )
            )
            self.enable_aug_cb.setChecked(
                self._safe_bool(state.get("enable_augmentation"), default=True)
            )
            self.aug_count_spin.setValue(
                self._safe_int(state.get("augment_times"), default=1, minimum=1, maximum=10)
            )
            self._set_aug_scope(str(state.get("augment_scope", "train")))

            selected_methods_raw = state.get("augment_methods")
            if isinstance(selected_methods_raw, list):
                selected_methods = {
                    str(item) for item in selected_methods_raw if isinstance(item, str)
                }
                for key, cb in self._method_checkboxes.items():
                    cb.setChecked(key in selected_methods)

            output_dir = state.get("output_dir")
            if isinstance(output_dir, str) and output_dir.strip():
                self.output_dir_edit.setText(output_dir.strip())

            self.skip_unlabeled_cb.setChecked(
                self._safe_bool(state.get("skip_unlabeled"), default=True)
            )
            self.overwrite_cb.setChecked(
                self._safe_bool(state.get("overwrite_output"), default=True)
            )

            self.custom_classes_cb.setChecked(
                self._safe_bool(state.get("use_custom_classes"), default=False)
            )
            custom_classes_file = state.get("custom_classes_file")
            if isinstance(custom_classes_file, str) and custom_classes_file.strip():
                self.custom_classes_edit.setText(custom_classes_file.strip())
        finally:
            self._restoring_ui_state = False

        self._update_ratio_hint()
        self._update_aug_hint()
        self._on_aug_toggled(self.enable_aug_cb.isChecked())
        self._on_custom_classes_toggled(self.custom_classes_cb.isChecked())

    def _save_ui_state(self) -> None:
        if self._restoring_ui_state:
            return
        self._save_state_timer.start()

    def _do_save_ui_state(self) -> None:
        state = {
            "project_id": self._current_project_id,
            "train_ratio": self.train_ratio_spin.value(),
            "random_seed": self.seed_spin.value(),
            "enable_augmentation": self.enable_aug_cb.isChecked(),
            "augment_times": self.aug_count_spin.value(),
            "augment_scope": self._get_aug_scope(),
            "augment_methods": self._selected_methods(),
            "output_dir": self.output_dir_edit.text().strip(),
            "skip_unlabeled": self.skip_unlabeled_cb.isChecked(),
            "overwrite_output": self.overwrite_cb.isChecked(),
            "use_custom_classes": self.custom_classes_cb.isChecked(),
            "custom_classes_file": self.custom_classes_edit.text().strip(),
        }
        try:
            self._ui_state_dir()
            with open(self._ui_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except OSError:
            return

    def _browse_output_dir(self):
        current = self.output_dir_edit.text().strip() or _get_default_output_dir()
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", current)
        if path:
            self.output_dir_edit.setText(path)

    def _on_custom_classes_toggled(self, enabled: bool):
        self.custom_classes_edit.setEnabled(enabled)
        self.browse_classes_btn.setEnabled(enabled)
        self.custom_classes_hint.setEnabled(enabled)

    def _browse_custom_classes(self):
        current = self.custom_classes_edit.text().strip()
        start_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择类别文件", start_dir, "文本文件 (*.txt);;所有文件 (*)"
        )
        if path:
            self.custom_classes_edit.setText(path)

    def _open_output_dir(self):
        path = self.output_dir_edit.text().strip()
        if path and os.path.isdir(path):
            open_path(path)

    def _on_start_clicked(self):
        if self._project_manager is None:
            return
        if self._current_project_id is None:
            InfoBar.warning(
                title="提示",
                content="请先选择数据集项目",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        project = self._project_manager.get_project(self._current_project_id)
        if not project:
            InfoBar.error(
                title="错误",
                content="数据集项目不存在",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        dirs = self._project_manager.get_directories(self._current_project_id)
        if not dirs:
            InfoBar.error(
                title="错误",
                content=f"数据集目录不存在: {project.directory}" if not project.is_archive_root
                        else "归档内没有有效目录",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            InfoBar.warning(
                title="提示",
                content="请选择输出目录",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        methods: List[str] = []
        augment_times = 0
        augment_scope = "train"
        if self.enable_aug_cb.isChecked():
            methods = self._selected_methods()
            if not methods:
                InfoBar.warning(
                    title="提示",
                    content="已启用增强，请至少选择一种增强方法",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                )
                return
            if not is_albumentations_available():
                InfoBar.error(
                    title="依赖缺失",
                    content="未安装 albumentations 库，无法进行数据增强。请先运行 pip install albumentations",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                    duration=5000,
                )
                return
            augment_times = self.aug_count_spin.value()
            augment_scope = self._get_aug_scope()

        custom_classes_file = None
        if self.custom_classes_cb.isChecked():
            custom_classes_file = self.custom_classes_edit.text().strip() or None
            if not custom_classes_file:
                InfoBar.warning(
                    title="提示",
                    content="已启用自定义类别文件，请选择 classes.txt 文件",
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                )
                return

        config = DataPrepConfig(
            dataset_name=project.name,
            dataset_dir=dirs[0] if dirs else "",
            output_dir=output_dir,
            train_ratio=self.train_ratio_spin.value() / 100.0,
            random_seed=self.seed_spin.value(),
            augment_methods=methods,
            augment_times=augment_times,
            augment_scope=augment_scope,
            skip_unlabeled=self.skip_unlabeled_cb.isChecked(),
            overwrite_output=self.overwrite_cb.isChecked(),
            custom_classes_file=custom_classes_file,
            dataset_dirs=dirs if len(dirs) > 1 else [],
        )

        self._worker = DataPrepWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_finished)

        self.progress_bar.setValue(0)
        self.progress_label.setText("正在准备数据...")
        self._set_running_state(True)
        self._run_started_at = perf_counter()
        self._last_progress_percent = -1
        self._last_progress_update_at = 0.0
        self._log_count = 0
        self._log("任务启动")
        self._worker.start()

    def _on_cancel_clicked(self):
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, percent: int, text: str):
        now = perf_counter()
        should_update = (
            percent != self._last_progress_percent
            or now - self._last_progress_update_at >= 0.2
        )
        if not should_update:
            return
        self._last_progress_percent = percent
        self._last_progress_update_at = now
        self.progress_bar.setValue(percent)
        self.progress_label.setText(text)

    def _on_finished(self, success: bool, message: str, summary_obj: object):
        self._set_running_state(False)
        if success:
            summary: DataPrepSummary = summary_obj
            self.progress_bar.setValue(100)
            self.progress_label.setText(
                f"完成: train={summary.train_images}, val={summary.val_images}, 增强={summary.augmented_images}"
            )
            self._log(
                f"完成: 输出 {summary.processed_images} 张, 类别 {summary.classes_count} 个, "
                f"YAML={summary.yaml_path}"
            )
            InfoBar.success(
                title="数据准备完成",
                content=f"输出目录: {summary.output_dir}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            self._log(f"失败: {message}")
            self.progress_label.setText("执行失败")
            InfoBar.error(
                title="执行失败",
                content=message,
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
        if self._run_started_at is not None:
            elapsed = perf_counter() - self._run_started_at
            self._log(f"运行耗时: {elapsed:.2f}s，日志条数: {self._log_count}")
            self._run_started_at = None
        self._worker = None

    def _set_running_state(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.dataset_combo.setEnabled(not running)
        self.refresh_dataset_btn.setEnabled(not running)
        self.train_ratio_spin.setEnabled(not running)
        self.seed_spin.setEnabled(not running)
        self.enable_aug_cb.setEnabled(not running)
        self.aug_count_spin.setEnabled(not running and self.enable_aug_cb.isChecked())
        self.aug_scope_combo.setEnabled(not running and self.enable_aug_cb.isChecked())
        self.aug_methods_container.setEnabled(not running and self.enable_aug_cb.isChecked())
        self.output_dir_edit.setEnabled(not running)
        self.browse_output_btn.setEnabled(not running)
        self.skip_unlabeled_cb.setEnabled(not running)
        self.overwrite_cb.setEnabled(not running)
        self.custom_classes_cb.setEnabled(not running)
        custom_enabled = not running and self.custom_classes_cb.isChecked()
        self.custom_classes_edit.setEnabled(custom_enabled)
        self.browse_classes_btn.setEnabled(custom_enabled)

    def _log(self, text: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_buffer.append(f"[{timestamp}] {text}")
        self._log_count += 1
        if not self._log_timer.isActive():
            self._log_timer.start()

    def _flush_log_buffer(self):
        if not self._log_buffer:
            self._log_timer.stop()
            return
        chunk = "\n".join(self._log_buffer)
        self._log_buffer.clear()
        self.log_edit.append(chunk)
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_edit.setTextCursor(cursor)
        self._log_timer.stop()

    def _clear_log(self):
        self._log_buffer.clear()
        self.log_edit.clear()
