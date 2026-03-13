"""预标注页面 - 视觉大模型预标注功能的用户界面"""

import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)
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
    PasswordLineEdit,
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
from ez_traing.prelabeling.config import APIConfigManager
from ez_traing.ui.workers import ImageScanWorker as ProjectImageScanWorker
from ez_traing.prelabeling.engine import PrelabelingWorker, validate_prelabeling_input
from ez_traing.prelabeling.models import DetectionMode, InferenceBackend, PrelabelingStats
from ez_traing.prelabeling.vision_service import VisionModelService
from ez_traing.prelabeling.yolo_service import YoloModelService

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "请检测图片中的所有目标物体，返回JSON格式的检测结果。\n"
    "返回格式要求：\n"
    '{"objects": [{"label": "类别名称", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.95}]}\n'
    "其中 bbox 坐标为像素值，label 为目标类别名称。"
)


class ReferenceImagePanel(QWidget):
    """参考图片管理面板

    允许用户添加、预览和管理参考图片，用于指导视觉模型进行目标检测。
    """

    images_changed = pyqtSignal(list)  # 参考图片列表变化时发出

    MAX_IMAGES = 10
    THUMBNAIL_SIZE = (80, 80)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_paths: List[str] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """设置参考图片面板布局"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 标题行：标题 + 计数标签
        header_layout = QHBoxLayout()
        header_layout.addWidget(StrongBodyLabel("参考图片", self))
        self._count_label = CaptionLabel("已添加 0/10 张参考图片", self)
        header_layout.addWidget(self._count_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # 图片列表
        self._list_widget = QListWidget(self)
        self._list_widget.setMinimumHeight(120)
        self._list_widget.setSpacing(4)
        self._list_widget.setIconSize(
            QSize(self.THUMBNAIL_SIZE[0], self.THUMBNAIL_SIZE[1])
        )
        layout.addWidget(self._list_widget)

        # 按钮行：添加 + 清空
        btn_layout = QHBoxLayout()
        self._add_btn = PushButton("添加参考图片", self)
        self._add_btn.setIcon(FIF.ADD)
        btn_layout.addWidget(self._add_btn)

        self._clear_btn = PushButton("清空全部", self)
        self._clear_btn.setIcon(FIF.DELETE)
        btn_layout.addWidget(self._clear_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Image management (implemented in tasks 2.2 - 2.4)
    # ------------------------------------------------------------------

    def add_images(self, paths: List[str]) -> List[str]:
        """添加参考图片，返回成功添加的路径列表

        对每张图片进行格式验证，跳过无效图片，检查最大数量限制，
        将有效图片添加到列表中。

        Args:
            paths: 待添加的图片文件路径列表

        Returns:
            成功添加的图片路径列表
        """
        added: List[str] = []

        for path in paths:
            # 检查是否已达到最大数量限制
            if len(self._image_paths) >= self.MAX_IMAGES:
                logger.warning("参考图片数量已达上限 %d 张", self.MAX_IMAGES)
                InfoBar.warning(
                    title="数量限制",
                    content=f"参考图片最多添加 {self.MAX_IMAGES} 张",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
                break

            # 验证图片格式
            if not self._validate_image(path):
                logger.warning("不支持的图片格式: %s", path)
                InfoBar.error(
                    title="格式不支持",
                    content=f"不支持的图片格式: {Path(path).name}",
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
                continue

            # 跳过已添加的图片
            if path in self._image_paths:
                continue

            # 添加到列表
            self._image_paths.append(path)
            added.append(path)

            # 在 UI 中显示图片项
            self._add_image_item(path)

        if added:
            self._update_count_label()
            self.images_changed.emit(list(self._image_paths))

        return added

    def remove_image(self, path: str) -> None:
        """移除指定参考图片

        从内部路径列表和列表控件中移除指定路径的图片，
        更新计数标签并发出 images_changed 信号。

        Args:
            path: 要移除的图片文件路径
        """
        if path not in self._image_paths:
            return

        self._image_paths.remove(path)

        # 在 _list_widget 中查找并移除对应项
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.data(Qt.UserRole) == path:
                self._list_widget.takeItem(i)
                break

        self._update_count_label()
        self.images_changed.emit(list(self._image_paths))

    def clear_all(self) -> None:
        """清空所有参考图片

        移除所有参考图片，清空列表控件，
        更新计数标签并发出 images_changed 信号。
        """
        self._image_paths.clear()
        self._list_widget.clear()
        self._update_count_label()
        self.images_changed.emit([])

    def get_image_paths(self) -> List[str]:
        """获取当前参考图片路径列表"""
        return list(self._image_paths)

    def get_image_count(self) -> int:
        """获取当前参考图片数量"""
        return len(self._image_paths)

    def _update_count_label(self) -> None:
        """更新计数标签显示"""
        count = len(self._image_paths)
        self._count_label.setText(f"已添加 {count}/{self.MAX_IMAGES} 张参考图片")

    # 参考图片支持的格式（需求 1.2 指定）
    SUPPORTED_REFERENCE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _validate_image(self, path: str) -> bool:
        """验证图片格式是否支持

        检查文件扩展名是否在支持的参考图片格式列表中。

        Args:
            path: 图片文件路径

        Returns:
            True 如果格式支持，否则 False
        """
        ext = Path(path).suffix.lower()
        return ext in self.SUPPORTED_REFERENCE_FORMATS

    def _create_thumbnail(self, path: str) -> QPixmap:
        """创建图片缩略图

        加载图片并缩放到 THUMBNAIL_SIZE，保持宽高比。
        如果图片无法加载，返回一个空的占位 QPixmap。

        Args:
            path: 图片文件路径

        Returns:
            缩放后的 QPixmap 缩略图
        """
        w, h = self.THUMBNAIL_SIZE
        pixmap = QPixmap(path)
        if pixmap.isNull():
            logger.warning("无法加载图片: %s", path)
            # 返回空白占位缩略图
            placeholder = QPixmap(w, h)
            placeholder.fill(Qt.lightGray)
            return placeholder
        return pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _add_image_item(self, path: str) -> None:
        """在列表中添加图片项

        创建带有缩略图和文件名的 QListWidgetItem 并添加到列表控件。

        Args:
            path: 图片文件路径
        """
        thumbnail = self._create_thumbnail(path)
        filename = Path(path).name

        item = QListWidgetItem()
        item.setIcon(QIcon(thumbnail))
        item.setText(filename)
        item.setToolTip(path)
        item.setData(Qt.UserRole, path)

        self._list_widget.addItem(item)


class PrelabelingPage(QWidget):
    """预标注页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_manager = APIConfigManager()
        self._worker: Optional[PrelabelingWorker] = None
        self._image_paths: List[str] = []
        self._project_manager = None  # ProjectManager reference
        self._current_project_id: Optional[str] = None
        self._project_ids: List[str] = []
        self._scan_worker: Optional[ProjectImageScanWorker] = None
        self._scan_cache: Dict[str, Tuple[int, List[str]]] = {}
        self._detection_mode: DetectionMode = DetectionMode.TEXT_ONLY
        self._inference_backend: InferenceBackend = InferenceBackend.YOLO_PT
        self._run_started_at: Optional[float] = None
        self._last_progress_percent = -1
        self._last_progress_text = ""
        self._last_progress_update_at = 0.0
        self._log_count = 0
        self._log_buffer: List[str] = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(100)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """设置页面布局：ScrollArea + 卡片"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content_widget = QWidget()
        self._content_layout = QVBoxLayout(content_widget)
        self._content_layout.setContentsMargins(36, 20, 36, 20)
        self._content_layout.setSpacing(16)

        # 页面标题
        title = TitleLabel("预标注", self)
        self._content_layout.addWidget(title)
        self._content_layout.addSpacing(4)

        # 各功能卡片
        self._content_layout.addWidget(self._create_dataset_card())
        self._config_card = self._create_config_card()
        self._content_layout.addWidget(self._config_card)
        self._prompt_card = self._create_prompt_card()
        self._content_layout.addWidget(self._prompt_card)
        self._content_layout.addWidget(self._create_action_card())
        self._content_layout.addWidget(self._create_log_card())

        self._content_layout.addStretch()
        self._apply_backend_ui_state()

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    # ------------------------------------------------------------------
    # Card: 数据集选择
    # ------------------------------------------------------------------

    def _create_dataset_card(self) -> CardWidget:
        """创建数据集选择器卡片"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("数据集选择", card))

        self.dataset_combo = ComboBox(card)
        self.dataset_combo.setPlaceholderText("请先在数据集页面创建项目")
        layout.addWidget(self.dataset_combo)

        self.dataset_info_label = CaptionLabel("", card)
        layout.addWidget(self.dataset_info_label)

        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)

        return card

    def _refresh_dataset_list(self) -> None:
        """刷新 ComboBox 中的数据集项目列表，保持当前选中项"""
        if self._project_manager is None:
            return

        previous_project_id = self._current_project_id

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self._project_ids.clear()

        projects = self._project_manager.get_all_projects(exclude_archived=True)
        for project in projects:
            self.dataset_combo.addItem(f"{project.name} ({project.image_count} 张图片)")
            self._project_ids.append(project.id)

        if not projects:
            self.dataset_info_label.setText("请先在数据集页面创建项目")
        elif previous_project_id in self._project_ids:
            idx = self._project_ids.index(previous_project_id)
            self.dataset_combo.setCurrentIndex(idx)
        else:
            self._current_project_id = None
            self._image_paths.clear()
            self.dataset_info_label.setText("")

        self.dataset_combo.blockSignals(False)

    def _on_dataset_changed(self, index: int) -> None:
        """ComboBox 选择变化时的回调，触发图片扫描"""
        if index < 0 or index >= len(self._project_ids):
            self._current_project_id = None
            self._image_paths.clear()
            self.dataset_info_label.setText("")
            return

        project_id = self._project_ids[index]
        self._current_project_id = project_id
        project = self._project_manager.get_project(project_id)
        if project:
            self._scan_project_images(project)

    def _scan_project_images(self, project) -> None:
        """扫描项目目录下的所有图片文件"""
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
            self._log(f"目录不存在: {project.directory}", level="error")
            InfoBar.error(
                title="错误",
                content=f"数据集目录不存在: {project.directory}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        if not project.is_archive_root:
            project_dir = Path(project.directory)
            try:
                directory_mtime = project_dir.stat().st_mtime_ns
            except OSError:
                directory_mtime = -1
            cache_key = project.id
            cached = self._scan_cache.get(cache_key)
            if cached is not None and cached[0] == directory_mtime:
                self._image_paths = list(cached[1])
                count = len(self._image_paths)
                self.dataset_info_label.setText(f"已加载 {count} 张图片（缓存）")
                self._log(f"数据集 '{project.name}' 读取缓存，共 {count} 张图片")
                return

        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()

        self.dataset_info_label.setText("正在扫描图片...")
        self._log(f"开始扫描数据集 '{project.name}' ...")
        if project.is_archive_root:
            self._scan_worker = ProjectImageScanWorker(
                project.id, directories=dirs)
        else:
            self._scan_worker = ProjectImageScanWorker(
                project.id, project.directory)
        self._scan_worker.finished.connect(self._on_project_scan_finished)
        self._scan_worker.start()

    def _on_project_scan_finished(
        self, project_id: str, image_paths: List[str], error: str, elapsed_sec: float
    ) -> None:
        if project_id != self._current_project_id:
            return

        project = self._project_manager.get_project(project_id) if self._project_manager else None
        if project is None:
            return

        if error:
            self._image_paths.clear()
            self.dataset_info_label.setText("扫描目录时出错")
            self._log(f"扫描目录出错: {error}", level="error")
            logger.error("扫描项目目录时出错: %s", error)
            return

        self._image_paths = list(image_paths)
        count = len(image_paths)
        self.dataset_info_label.setText(f"已加载 {count} 张图片")
        self._log(f"数据集 '{project.name}' 已加载 {count} 张图片，耗时 {elapsed_sec:.2f}s")

        try:
            directory_mtime = Path(project.directory).stat().st_mtime_ns
        except OSError:
            directory_mtime = -1
        self._scan_cache[project_id] = (directory_mtime, list(image_paths))


    # ------------------------------------------------------------------
    # Card: API 配置
    # ------------------------------------------------------------------

    def _create_config_card(self) -> CardWidget:
        """创建 API 配置卡片"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("推理配置", card))

        layout.addWidget(StrongBodyLabel("推理方式", card))
        self.inference_backend_combo = ComboBox(card)
        self.inference_backend_combo.addItem("视觉 API")
        self.inference_backend_combo.addItem("本地 YOLO (.pt)")
        self.inference_backend_combo.setCurrentIndex(1)
        self.inference_backend_combo.currentIndexChanged.connect(
            self._on_inference_backend_changed
        )
        layout.addWidget(self.inference_backend_combo)

        self.backend_hint_label = CaptionLabel("", card)
        self.backend_hint_label.setWordWrap(True)
        layout.addWidget(self.backend_hint_label)

        # API 配置区域
        self.api_config_widget = QWidget(card)
        api_layout = QVBoxLayout(self.api_config_widget)
        api_layout.setContentsMargins(0, 0, 0, 0)
        api_layout.setSpacing(10)
        api_layout.addWidget(StrongBodyLabel("API 地址", card))
        self.endpoint_edit = LineEdit(card)
        self.endpoint_edit.setPlaceholderText("https://api.openai.com/v1/chat/completions")
        api_layout.addWidget(self.endpoint_edit)

        api_layout.addWidget(StrongBodyLabel("API 令牌", card))
        self.api_key_edit = PasswordLineEdit(card)
        self.api_key_edit.setPlaceholderText("sk-...")
        api_layout.addWidget(self.api_key_edit)

        api_layout.addWidget(StrongBodyLabel("模型名称", card))
        self.model_name_edit = LineEdit(card)
        self.model_name_edit.setPlaceholderText("gpt-4-vision-preview")
        api_layout.addWidget(self.model_name_edit)

        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(StrongBodyLabel("超时时间 (秒)", card))
        self.timeout_spin = SpinBox(card)
        self.timeout_spin.setRange(10, 600)
        self.timeout_spin.setValue(60)
        timeout_layout.addWidget(self.timeout_spin)
        timeout_layout.addStretch()
        api_layout.addLayout(timeout_layout)

        self.save_config_btn = PushButton("保存配置", card)
        self.save_config_btn.setIcon(FIF.SAVE)
        self.save_config_btn.clicked.connect(self._on_save_config)
        api_layout.addWidget(self.save_config_btn)
        layout.addWidget(self.api_config_widget)

        # YOLO 本地配置区域
        self.yolo_config_widget = QWidget(card)
        yolo_layout = QVBoxLayout(self.yolo_config_widget)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        yolo_layout.setSpacing(8)
        yolo_layout.addWidget(StrongBodyLabel("YOLO 权重文件 (.pt)", card))
        yolo_row = QHBoxLayout()
        self.yolo_model_edit = LineEdit(card)
        self.yolo_model_edit.setPlaceholderText("选择训练好的 .pt 文件")
        yolo_row.addWidget(self.yolo_model_edit)
        self.browse_yolo_model_btn = PushButton("浏览", card)
        self.browse_yolo_model_btn.setIcon(FIF.FOLDER)
        self.browse_yolo_model_btn.clicked.connect(self._browse_yolo_model)
        yolo_row.addWidget(self.browse_yolo_model_btn)
        yolo_layout.addLayout(yolo_row)
        yolo_layout.addWidget(
            CaptionLabel("将使用该权重对数据集图片进行目标识别并生成 VOC 标注", card)
        )
        layout.addWidget(self.yolo_config_widget)

        # 加载已有配置到 UI
        self._load_config_to_ui()

        return card

    def _load_config_to_ui(self) -> None:
        """将已保存的配置加载到输入框"""
        cfg = self._config_manager.get_config()
        self.endpoint_edit.setText(cfg.endpoint)
        self.api_key_edit.setText(cfg.api_key)
        self.model_name_edit.setText(cfg.model_name)
        self.timeout_spin.setValue(cfg.timeout)

    def _on_save_config(self) -> None:
        """保存 API 配置"""
        self._config_manager.update_config(
            endpoint=self.endpoint_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            model_name=self.model_name_edit.text().strip() or "gpt-4-vision-preview",
            timeout=self.timeout_spin.value(),
        )
        self._log("API 配置已保存")
        InfoBar.success(
            title="保存成功",
            content="API 配置已保存",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2000,
        )

    def _browse_yolo_model(self) -> None:
        """选择 YOLO .pt 权重文件。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 权重文件",
            "",
            "PyTorch 权重 (*.pt);;所有文件 (*)",
        )
        if path:
            self.yolo_model_edit.setText(path)

    def _on_inference_backend_changed(self, index: int) -> None:
        """切换推理后端。"""
        self._inference_backend = (
            InferenceBackend.VISION_API if index == 0 else InferenceBackend.YOLO_PT
        )
        self._apply_backend_ui_state()

    def _apply_backend_ui_state(self) -> None:
        """根据推理后端切换 UI 可见状态。"""
        is_api = self._inference_backend == InferenceBackend.VISION_API
        self.api_config_widget.setVisible(is_api)
        self.yolo_config_widget.setVisible(not is_api)
        if hasattr(self, "_prompt_card"):
            self._prompt_card.setVisible(is_api)
        if is_api:
            self.backend_hint_label.setText("使用远程视觉 API 进行预标注")
            if hasattr(self, "start_btn"):
                self.start_btn.setText("开始预标注")
        else:
            self.backend_hint_label.setText("使用本地 YOLO .pt 模型进行识别并生成/合并 VOC 标注")
            if hasattr(self, "start_btn"):
                self.start_btn.setText("开始识别")

    # ------------------------------------------------------------------
    # Card: 提示词
    # ------------------------------------------------------------------

    def _create_prompt_card(self) -> CardWidget:
        """创建提示词输入卡片"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("提示词", card))

        # 检测模式选择器
        layout.addWidget(StrongBodyLabel("检测模式", card))
        self.mode_combo = ComboBox(card)
        self.mode_combo.addItem("仅文本提示")
        self.mode_combo.addItem("参考图片")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        layout.addWidget(CaptionLabel("描述您希望模型检测的目标内容", card))

        self.prompt_edit = TextEdit(card)
        self.prompt_edit.setPlaceholderText("输入提示词...")
        self.prompt_edit.setText(DEFAULT_PROMPT)
        self.prompt_edit.setMinimumHeight(120)
        layout.addWidget(self.prompt_edit)

        # 参考图片面板
        self._ref_panel = ReferenceImagePanel(card)
        self._ref_panel.setVisible(False)
        layout.addWidget(self._ref_panel)

        # 连接参考图片面板按钮
        self._ref_panel._add_btn.clicked.connect(self._on_add_reference_images)
        self._ref_panel._clear_btn.clicked.connect(self._ref_panel.clear_all)
        self._ref_panel.images_changed.connect(self._on_reference_images_changed)

        return card

    def _on_mode_changed(self, index: int) -> None:
        """检测模式切换回调"""
        if index == 0:
            self._detection_mode = DetectionMode.TEXT_ONLY
            self._ref_panel.setVisible(False)
        else:
            self._detection_mode = DetectionMode.REFERENCE_IMAGE
            self._ref_panel.setVisible(True)

    def _on_add_reference_images(self) -> None:
        """打开文件对话框选择参考图片"""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择参考图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        if paths:
            self._ref_panel.add_images(paths)

    def _on_reference_images_changed(self, paths: list) -> None:
        """参考图片列表变化时记录日志"""
        self._log(f"参考图片已更新，当前数量: {len(paths)}")

    # ------------------------------------------------------------------
    # Card: 操作按钮
    # ------------------------------------------------------------------

    def _create_action_card(self) -> CardWidget:
        """创建操作按钮卡片"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("操作", card))

        # 复选框
        self.skip_annotated_cb = CheckBox("仅识别未标注图片（跳过已有 XML）", card)
        self.skip_annotated_cb.setChecked(True)
        layout.addWidget(self.skip_annotated_cb)
        layout.addWidget(
            CaptionLabel(
                "取消勾选后将识别全部图片；若图片已有 XML 标注，会与识别结果合并保存。",
                card,
            )
        )

        # 并发线程数
        concurrency_layout = QHBoxLayout()
        concurrency_layout.addWidget(StrongBodyLabel("并发线程数", card))
        self.concurrency_spin = SpinBox(card)
        self.concurrency_spin.setRange(1, 16)
        self.concurrency_spin.setValue(1)
        self.concurrency_spin.setToolTip("同时发送的 API 请求数量，增大可加速处理但会增加 API 负载")
        concurrency_layout.addWidget(self.concurrency_spin)
        concurrency_layout.addStretch()
        layout.addLayout(concurrency_layout)

        # 按钮行
        btn_layout = QHBoxLayout()
        self.start_btn = PrimaryPushButton("开始预标注", card)
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = PushButton("取消", card)
        self.cancel_btn.setIcon(FIF.CLOSE)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = ProgressBar(card)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = CaptionLabel("就绪", card)
        layout.addWidget(self.progress_label)

        return card

    # ------------------------------------------------------------------
    # Card: 日志
    # ------------------------------------------------------------------

    def _create_log_card(self) -> CardWidget:
        """创建日志显示卡片"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.addWidget(SubtitleLabel("日志", card))
        header_layout.addStretch()
        clear_btn = PushButton("清空", card)
        clear_btn.clicked.connect(self._clear_log)
        header_layout.addWidget(clear_btn)
        layout.addLayout(header_layout)

        self.log_text = TextEdit(card)
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

        return card

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        """添加带时间戳的日志条目"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        tag = level.upper()
        self._log_buffer.append(f"[{timestamp}] [{tag}] {message}")
        self._log_count += 1
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    _MAX_LOG_LINES = 5000

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            self._log_flush_timer.stop()
            return
        chunk = "\n".join(self._log_buffer)
        self._log_buffer.clear()
        self.log_text.append(chunk)

        doc = self.log_text.document()
        if doc.blockCount() > self._MAX_LOG_LINES:
            cursor = QTextCursor(doc)
            cursor.movePosition(QTextCursor.Start)
            excess = doc.blockCount() - self._MAX_LOG_LINES
            for _ in range(excess):
                cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor)
            cursor.movePosition(
                QTextCursor.StartOfBlock, QTextCursor.KeepAnchor,
            )
            cursor.removeSelectedText()
            cursor.deleteChar()

        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        self._log_flush_timer.stop()

    def _clear_log(self) -> None:
        self._log_buffer.clear()
        self.log_text.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project_manager(self, manager) -> None:
        """设置项目管理器引用（由 AppWindow 调用）"""
        self._project_manager = manager

    def showEvent(self, event) -> None:
        """重写 showEvent，页面可见时刷新数据集列表"""
        super().showEvent(event)
        self._refresh_dataset_list()

    def set_image_paths(self, paths: List[str]) -> None:
        """从外部设置待处理的图片路径列表"""
        self._image_paths = list(paths)
        self._log(f"已加载 {len(self._image_paths)} 张图片")

    # ------------------------------------------------------------------
    # Flow control
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        """开始预标注"""
        prompt = self.prompt_edit.toPlainText().strip()

        # 参考图片模式验证：仅视觉 API 模式下需要
        if (
            self._inference_backend == InferenceBackend.VISION_API
            and self._detection_mode == DetectionMode.REFERENCE_IMAGE
            and not self._ref_panel.get_image_paths()
        ):
            InfoBar.warning(
                title="提示",
                content="请先添加至少一张参考图片",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        # 获取参考图片列表（仅参考图片模式下传递）
        reference_images = None
        if (
            self._inference_backend == InferenceBackend.VISION_API
            and self._detection_mode == DetectionMode.REFERENCE_IMAGE
        ):
            reference_images = self._ref_panel.get_image_paths()

        # 验证输入
        try:
            validate_prelabeling_input(
                prompt,
                self._config_manager,
                inference_backend=self._inference_backend.value,
                yolo_model_path=self.yolo_model_edit.text().strip(),
                detection_mode=self._detection_mode.value,
                reference_images=reference_images,
            )
        except ValueError as e:
            InfoBar.warning(
                title="提示",
                content=str(e),
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        if self._current_project_id is None:
            InfoBar.warning(
                title="提示",
                content="请先选择数据集",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        if not self._image_paths:
            InfoBar.warning(
                title="提示",
                content="当前数据集没有可处理的图片",
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
            return

        # 创建服务和工作线程
        vision_service = None
        yolo_service = None
        max_workers = self.concurrency_spin.value()
        if self._inference_backend == InferenceBackend.VISION_API:
            vision_service = VisionModelService(self._config_manager)
        else:
            try:
                yolo_service = YoloModelService(
                    model_path=self.yolo_model_edit.text().strip()
                )
            except Exception as e:
                InfoBar.error(
                    title="加载模型失败",
                    content=str(e),
                    parent=self.window(),
                    position=InfoBarPosition.TOP,
                )
                return
            # 本地模型推理使用单线程，避免模型并发访问造成不稳定
            if max_workers > 1:
                self._log("本地 YOLO 模式下并发已自动调整为 1", level="warning")
                max_workers = 1

        self._worker = PrelabelingWorker(
            image_paths=self._image_paths,
            prompt=prompt,
            vision_service=vision_service,
            yolo_service=yolo_service,
            inference_backend=self._inference_backend.value,
            skip_annotated=self.skip_annotated_cb.isChecked(),
            max_workers=max_workers,
            reference_images=reference_images,
            detection_mode=self._detection_mode.value,
        )

        # 连接信号
        self._worker.progress.connect(self._on_progress)
        self._worker.image_completed.connect(self._on_image_completed)
        self._worker.finished.connect(self._on_finished)

        # 更新 UI 状态
        self._set_running_state(True)
        self._run_started_at = perf_counter()
        self._last_progress_percent = -1
        self._last_progress_text = ""
        self._last_progress_update_at = 0.0
        self._log_count = 0
        self._log("预标注开始")

        # 记录检测模式和参考图片信息
        if self._inference_backend == InferenceBackend.VISION_API:
            mode_label = (
                "参考图片"
                if self._detection_mode == DetectionMode.REFERENCE_IMAGE
                else "仅文本提示"
            )
            self._log("推理方式: 视觉 API")
            self._log(f"检测模式: {mode_label}")
            if reference_images:
                self._log(f"参考图片数量: {len(reference_images)}")
        else:
            self._log("推理方式: 本地 YOLO")
            self._log(f"权重文件: {self.yolo_model_edit.text().strip()}")

        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        """取消预标注"""
        if self._worker:
            self._worker.cancel()
            self._log("正在取消...", level="warning")

    def _on_progress(self, current: int, total: int, message: str) -> None:
        """处理进度更新"""
        if total > 0:
            percent = int(current / total * 100)
            now = perf_counter()
            should_update = (
                percent != self._last_progress_percent
                or now - self._last_progress_update_at >= 0.2
            )
            if not should_update:
                return
            self._last_progress_percent = percent
            self._last_progress_text = message
            self._last_progress_update_at = now
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"{current}/{total} - {message}")

    def _on_image_completed(self, path: str, success: bool, message: str) -> None:
        """单张图片处理完成"""
        level = "info" if success else "error"
        self._log(message, level=level)

    def _on_finished(self, stats: PrelabelingStats) -> None:
        """批量处理完成"""
        self._set_running_state(False)
        self.progress_bar.setValue(100)

        not_processed = stats.total - stats.processed - stats.skipped
        summary = (
            f"预标注完成 - 总计: {stats.total}, "
            f"成功: {stats.success}, 失败: {stats.failed}, "
            f"跳过: {stats.skipped}"
        )
        if not_processed > 0:
            summary += f", 未处理: {not_processed}"
        self._log(summary)
        if self._run_started_at is not None:
            elapsed = perf_counter() - self._run_started_at
            self._log(f"运行耗时: {elapsed:.2f}s，日志条数: {self._log_count}")
            self._run_started_at = None
        self.progress_label.setText(summary)

        InfoBar.success(
            title="预标注完成",
            content=summary,
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=5000,
        )
        self._worker = None

    def _set_running_state(self, running: bool) -> None:
        """切换运行/就绪 UI 状态"""
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.cancel_btn.setVisible(running)
        self.skip_annotated_cb.setEnabled(not running)
        self.save_config_btn.setEnabled(not running)
        self.dataset_combo.setEnabled(not running)
        self.concurrency_spin.setEnabled(not running)
        self.inference_backend_combo.setEnabled(not running)
        self.browse_yolo_model_btn.setEnabled(not running)
        self.yolo_model_edit.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
