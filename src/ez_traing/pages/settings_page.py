"""设置页面 - 显示环境信息、GPU 状态和应用更新"""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QLabel,
    QHBoxLayout,
    QApplication,
    QMessageBox,
    QFileDialog,
)
from qfluentwidgets import (
    CardWidget,
    FluentIcon as FIF,
    IconWidget,
    BodyLabel,
    CaptionLabel,
    StrongBodyLabel,
    ScrollArea,
    TitleLabel,
    InfoBar,
    InfoBarPosition,
    PushButton,
    PrimaryPushButton,
    ProgressBar,
    MessageBox,
    SwitchButton,
    ComboBox,
    LineEdit,
)

from ez_traing import __version__
from ez_traing.common.constants import load_settings, save_settings
from ez_traing.updater import (
    is_frozen,
    CheckUpdateWorker,
    DownloadWorker,
    apply_update_and_restart,
)


def _get_package_info():
    """安全获取包版本信息，未安装时返回 None"""
    info = {
        "ultralytics_version": None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_names": [],
    }

    # 检测 ultralytics
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        pass

    # 检测 torch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if info["cuda_available"] else None
        if info["cuda_available"]:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                info["gpu_names"].append(torch.cuda.get_device_name(i))
    except ImportError:
        pass

    return info


class PackageInstallCard(CardWidget):
    """包安装建议卡片"""

    def __init__(self, title: str, description: str, install_cmd: str, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # 标题
        title_layout = QHBoxLayout()
        icon_widget = IconWidget(FIF.DOWNLOAD, self)
        icon_widget.setFixedSize(20, 20)
        title_label = StrongBodyLabel(title, self)
        title_layout.addWidget(icon_widget)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # 说明
        desc_label = CaptionLabel(description, self)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # 安装命令
        cmd_widget = self._create_command_widget(install_cmd)
        layout.addWidget(cmd_widget)

    def _create_command_widget(self, command: str) -> QWidget:
        """创建命令显示组件，带复制按钮"""
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(8)

        cmd_label = CaptionLabel(command, widget)
        cmd_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.05); "
            "padding: 8px; "
            "border-radius: 4px; "
            "font-family: Consolas, monospace;"
        )
        cmd_label.setWordWrap(True)
        layout.addWidget(cmd_label, 1)

        copy_btn = PushButton("复制", widget)
        copy_btn.setFixedWidth(60)
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(command))
        layout.addWidget(copy_btn)

        return widget

    def _copy_to_clipboard(self, text: str):
        """复制文本到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        InfoBar.success(
            title="已复制",
            content="安装命令已复制到剪贴板",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self.window(),
        )


class InfoCard(CardWidget):
    """信息展示卡片"""

    def __init__(self, icon, title, content, parent=None):
        super().__init__(parent)
        self.setFixedHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)

        # 图标
        icon_widget = IconWidget(icon, self)
        icon_widget.setFixedSize(32, 32)
        layout.addWidget(icon_widget)

        # 文本区域
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)

        title_label = BodyLabel(title, self)
        title_label.setObjectName("cardTitle")

        self.content_label = StrongBodyLabel(content, self)
        self.content_label.setObjectName("cardContent")

        text_layout.addWidget(title_label)
        text_layout.addWidget(self.content_label)
        layout.addLayout(text_layout)
        layout.addStretch()

    def set_content(self, content: str):
        self.content_label.setText(content)


class StatusCard(CardWidget):
    """状态展示卡片，带状态指示"""

    def __init__(self, icon, title, status_ok: bool, content: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)

        # 图标
        icon_widget = IconWidget(icon, self)
        icon_widget.setFixedSize(32, 32)
        layout.addWidget(icon_widget)

        # 文本区域
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)

        title_label = BodyLabel(title, self)

        # 状态内容
        status_text = content
        self.content_label = StrongBodyLabel(status_text, self)
        if status_ok:
            self.content_label.setStyleSheet("color: #0f9d58;")  # 绿色
        else:
            self.content_label.setStyleSheet("color: #db4437;")  # 红色

        text_layout.addWidget(title_label)
        text_layout.addWidget(self.content_label)
        layout.addLayout(text_layout)
        layout.addStretch()


class InstallSuggestionCard(CardWidget):
    """安装建议卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # 标题
        title_layout = QHBoxLayout()
        icon_widget = IconWidget(FIF.INFO, self)
        icon_widget.setFixedSize(20, 20)
        title_label = StrongBodyLabel("GPU 版本安装建议", self)
        title_layout.addWidget(icon_widget)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # 说明
        desc_label = CaptionLabel(
            "当前安装的 PyTorch 不支持 GPU 加速。如需使用 GPU 训练，请先卸载当前版本，"
            "然后根据您的 CUDA 版本选择合适的安装命令：",
            self,
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # CUDA 11.8 安装命令
        cuda118_label = CaptionLabel("CUDA 11.8:", self)
        layout.addWidget(cuda118_label)

        cuda118_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        self.cuda118_widget = self._create_command_widget(cuda118_cmd)
        layout.addWidget(self.cuda118_widget)

        # CUDA 12.1 安装命令
        cuda121_label = CaptionLabel("CUDA 12.1 / 12.4:", self)
        layout.addWidget(cuda121_label)

        cuda121_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        self.cuda121_widget = self._create_command_widget(cuda121_cmd)
        layout.addWidget(self.cuda121_widget)

        # 提示
        tip_label = CaptionLabel(
            "提示：安装前请先运行 pip uninstall torch torchvision torchaudio 卸载现有版本。",
            self,
        )
        tip_label.setStyleSheet("color: #f4b400;")
        layout.addWidget(tip_label)

    def _create_command_widget(self, command: str) -> QWidget:
        """创建命令显示组件，带复制按钮"""
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(8)

        cmd_label = CaptionLabel(command, widget)
        cmd_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.05); "
            "padding: 8px; "
            "border-radius: 4px; "
            "font-family: Consolas, monospace;"
        )
        cmd_label.setWordWrap(True)
        layout.addWidget(cmd_label, 1)

        copy_btn = PushButton("复制", widget)
        copy_btn.setFixedWidth(60)
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(command))
        layout.addWidget(copy_btn)

        return widget

    def _copy_to_clipboard(self, text: str):
        """复制文本到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        InfoBar.success(
            title="已复制",
            content="安装命令已复制到剪贴板",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self.window(),
        )


class SettingsPage(QWidget):
    """设置页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """设置界面"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 滚动区域
        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # 内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(36, 20, 36, 20)
        content_layout.setSpacing(16)

        # 页面标题
        title_label = TitleLabel("设置", self)
        content_layout.addWidget(title_label)
        content_layout.addSpacing(10)

        # ── 应用版本 ──────────────────────────────────────────────
        app_group_label = StrongBodyLabel("应用版本", self)
        content_layout.addWidget(app_group_label)

        self._version_card = self._create_update_card(content_layout)
        content_layout.addSpacing(10)

        # ── GitHub 加速 ──────────────────────────────────────────────
        mirror_group_label = StrongBodyLabel("GitHub 加速", self)
        content_layout.addWidget(mirror_group_label)

        self._create_mirror_card(content_layout)
        content_layout.addSpacing(10)

        # ── 加样设置 ──────────────────────────────────────────────
        sample_group_label = StrongBodyLabel("加样设置", self)
        content_layout.addWidget(sample_group_label)

        self._create_sample_dir_card(content_layout)
        content_layout.addSpacing(10)

        # 获取版本信息
        pkg_info = _get_package_info()
        ultralytics_installed = pkg_info["ultralytics_version"] is not None
        torch_installed = pkg_info["torch_version"] is not None
        cuda_available = pkg_info["cuda_available"]

        # 环境信息分组标签
        env_group_label = StrongBodyLabel("环境信息", self)
        content_layout.addWidget(env_group_label)

        # Ultralytics 版本卡片
        ultralytics_version_text = pkg_info["ultralytics_version"] if ultralytics_installed else "未安装"
        ultralytics_card = InfoCard(
            FIF.APPLICATION, "Ultralytics 版本", ultralytics_version_text, self
        )
        if not ultralytics_installed:
            ultralytics_card.content_label.setStyleSheet("color: #db4437;")
        content_layout.addWidget(ultralytics_card)

        # PyTorch 版本卡片
        torch_version_text = pkg_info["torch_version"] if torch_installed else "未安装"
        torch_card = InfoCard(FIF.DEVELOPER_TOOLS, "PyTorch 版本", torch_version_text, self)
        if not torch_installed:
            torch_card.content_label.setStyleSheet("color: #db4437;")
        content_layout.addWidget(torch_card)

        content_layout.addSpacing(10)

        # GPU 状态分组标签
        gpu_group_label = StrongBodyLabel("GPU 状态", self)
        content_layout.addWidget(gpu_group_label)

        # CUDA 可用状态
        if not torch_installed:
            cuda_status_text = "未知 (需先安装 PyTorch)"
            cuda_status_ok = False
        else:
            cuda_status_text = "已启用" if cuda_available else "未启用"
            cuda_status_ok = cuda_available

        cuda_status_card = StatusCard(
            FIF.SPEED_HIGH if cuda_status_ok else FIF.SPEED_OFF,
            "CUDA 加速",
            cuda_status_ok,
            cuda_status_text,
            self,
        )
        content_layout.addWidget(cuda_status_card)

        # CUDA 版本
        if not torch_installed:
            cuda_version_text = "未知"
        else:
            cuda_version_text = pkg_info["cuda_version"] if pkg_info["cuda_version"] else "不可用"
        cuda_version_card = InfoCard(FIF.TAG, "CUDA 版本", cuda_version_text, self)
        content_layout.addWidget(cuda_version_card)

        # GPU 设备信息
        if not torch_installed:
            gpu_info = "未知 (需先安装 PyTorch)"
        elif cuda_available and pkg_info["gpu_names"]:
            gpu_info = ", ".join(pkg_info["gpu_names"])
        else:
            gpu_info = "未检测到 (需要 CUDA 支持)"

        gpu_device_card = InfoCard(FIF.ROBOT, "GPU 设备", gpu_info, self)
        content_layout.addWidget(gpu_device_card)

        # 安装建议部分
        need_suggestion = not ultralytics_installed or not torch_installed or not cuda_available
        if need_suggestion:
            content_layout.addSpacing(10)
            suggestion_group_label = StrongBodyLabel("安装建议", self)
            content_layout.addWidget(suggestion_group_label)

            # Ultralytics 未安装
            if not ultralytics_installed:
                ultralytics_install_card = PackageInstallCard(
                    "安装 Ultralytics",
                    "Ultralytics 是 YOLO 模型的官方实现库，用于目标检测模型的训练和推理。",
                    "pip install ultralytics",
                    self,
                )
                content_layout.addWidget(ultralytics_install_card)

            # PyTorch 未安装
            if not torch_installed:
                torch_install_card = PackageInstallCard(
                    "安装 PyTorch (GPU 版本)",
                    "PyTorch 是深度学习框架，建议安装支持 GPU 的版本以加速训练。"
                    "请根据您的 CUDA 版本选择合适的安装命令。",
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                    self,
                )
                content_layout.addWidget(torch_install_card)
            # PyTorch 已安装但不支持 GPU
            elif not cuda_available:
                suggestion_card = InstallSuggestionCard(self)
                content_layout.addWidget(suggestion_card)

        # 弹性空间
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

    # ── GitHub 加速设置 ────────────────────────────────────────────

    _PRESET_MIRRORS = [
        ("ghp.ci", "https://ghp.ci/"),
        ("mirror.ghproxy.com", "https://mirror.ghproxy.com/"),
        ("gh-proxy.com", "https://gh-proxy.com/"),
        ("自定义", ""),
    ]

    def _create_mirror_card(self, parent_layout: QVBoxLayout) -> None:
        settings = load_settings()
        enabled = settings.get("github_mirror_enabled", False)
        saved_url = settings.get("github_mirror_url", self._PRESET_MIRRORS[0][1])

        card = CardWidget(self)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        icon_widget = IconWidget(FIF.GLOBE, self)
        icon_widget.setFixedSize(32, 32)
        top_row.addWidget(icon_widget)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)
        title_label = BodyLabel("GitHub 加速站", self)
        desc_label = CaptionLabel(
            "国内网络访问 GitHub 较慢时，可启用加速站代理下载和更新检查",
            self,
        )
        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)
        top_row.addLayout(text_layout)
        top_row.addStretch()

        self._mirror_switch = SwitchButton(self)
        self._mirror_switch.setChecked(enabled)
        top_row.addWidget(self._mirror_switch)
        card_layout.addLayout(top_row)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(12)
        combo_label = BodyLabel("加速站地址", self)
        combo_row.addWidget(combo_label)

        self._mirror_combo = ComboBox(self)
        for display_name, _ in self._PRESET_MIRRORS:
            self._mirror_combo.addItem(display_name)
        self._mirror_combo.setMinimumWidth(200)

        preset_index = self._find_preset_index(saved_url)
        self._mirror_combo.setCurrentIndex(preset_index)

        combo_row.addWidget(self._mirror_combo)
        combo_row.addStretch()
        card_layout.addLayout(combo_row)

        self._custom_edit = LineEdit(self)
        self._custom_edit.setPlaceholderText("请输入加速站 URL，如 https://example.com/")
        is_custom = preset_index == len(self._PRESET_MIRRORS) - 1
        if is_custom:
            self._custom_edit.setText(saved_url)
        self._custom_edit.setVisible(is_custom)

        card_layout.addWidget(self._custom_edit)

        self._mirror_combo.setEnabled(enabled)
        self._custom_edit.setEnabled(enabled)

        self._mirror_switch.checkedChanged.connect(self._on_mirror_toggled)
        self._mirror_combo.currentIndexChanged.connect(self._on_mirror_combo_changed)
        self._custom_edit.editingFinished.connect(self._on_custom_mirror_edited)

        parent_layout.addWidget(card)

    def _find_preset_index(self, url: str) -> int:
        for i, (_, preset_url) in enumerate(self._PRESET_MIRRORS[:-1]):
            if url == preset_url:
                return i
        return len(self._PRESET_MIRRORS) - 1

    def _on_mirror_toggled(self, checked):
        self._mirror_combo.setEnabled(checked)
        is_custom = self._mirror_combo.currentIndex() == len(self._PRESET_MIRRORS) - 1
        self._custom_edit.setEnabled(checked)
        self._custom_edit.setVisible(checked and is_custom)
        self._save_mirror_settings()

    def _on_mirror_combo_changed(self, index: int):
        is_custom = index == len(self._PRESET_MIRRORS) - 1
        self._custom_edit.setVisible(is_custom)
        if is_custom:
            self._custom_edit.setFocus()
        else:
            self._save_mirror_settings()

    def _on_custom_mirror_edited(self):
        self._save_mirror_settings()

    def _save_mirror_settings(self):
        enabled = self._mirror_switch.isChecked()
        index = self._mirror_combo.currentIndex()
        is_custom = index == len(self._PRESET_MIRRORS) - 1
        if is_custom:
            url = self._custom_edit.text().strip()
        else:
            url = self._PRESET_MIRRORS[index][1]

        if enabled and is_custom:
            if not url:
                self._mirror_switch.setChecked(False)
                InfoBar.warning(
                    title="加速站地址为空",
                    content="请先输入加速站地址",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self.window(),
                )
                return
            if not url.startswith(("http://", "https://")):
                InfoBar.warning(
                    title="地址格式不正确",
                    content="加速站地址需以 http:// 或 https:// 开头",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self.window(),
                )
                return

        settings = load_settings()
        settings["github_mirror_enabled"] = enabled
        settings["github_mirror_url"] = url
        save_settings(settings)

    # ── 更新功能 ──────────────────────────────────────────────────

    def _create_update_card(self, parent_layout: QVBoxLayout) -> CardWidget:
        card = CardWidget(self)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(16)

        icon_widget = IconWidget(FIF.UPDATE, self)
        icon_widget.setFixedSize(32, 32)
        layout.addWidget(icon_widget)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)

        title_label = BodyLabel("当前版本", self)
        mode = "开发模式" if not is_frozen() else ""
        version_text = f"v{__version__}  {mode}".strip()
        self._version_label = StrongBodyLabel(version_text, self)

        text_layout.addWidget(title_label)
        text_layout.addWidget(self._version_label)
        layout.addLayout(text_layout)
        layout.addStretch()

        self._update_btn = PrimaryPushButton("检查更新", self)
        self._update_btn.setFixedWidth(110)
        self._update_btn.clicked.connect(self._on_check_update)
        layout.addWidget(self._update_btn)

        parent_layout.addWidget(card)

        self._progress_bar = ProgressBar(self)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.hide()
        parent_layout.addWidget(self._progress_bar)

        self._check_worker = None
        self._download_worker = None

        return card

    def _on_check_update(self):
        self._update_btn.setEnabled(False)
        self._update_btn.setText("检查中...")

        self._check_worker = CheckUpdateWorker(self)
        self._check_worker.finished.connect(self._on_check_finished)
        self._check_worker.error.connect(self._on_check_error)
        self._check_worker.start()

    def _on_check_finished(self, release_info):
        self._update_btn.setEnabled(True)
        self._update_btn.setText("检查更新")

        if release_info is None:
            InfoBar.success(
                title="已是最新版本",
                content=f"当前版本 v{__version__} 已是最新",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self.window(),
            )
            return

        body_preview = release_info.body[:300] if release_info.body else ""
        size_mb = f"{release_info.size / 1024 / 1024:.1f} MB" if release_info.size else ""

        dlg = MessageBox(
            f"发现新版本 {release_info.tag}",
            f"{body_preview}\n\n文件大小: {size_mb}" if size_mb else body_preview,
            self.window(),
        )
        dlg.yesButton.setText("立即更新")
        dlg.cancelButton.setText("稍后")

        if dlg.exec():
            self._start_download(release_info)

    def _on_check_error(self, msg: str):
        self._update_btn.setEnabled(True)
        self._update_btn.setText("检查更新")
        InfoBar.error(
            title="检查更新失败",
            content=msg,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self.window(),
        )

    def _start_download(self, release_info):
        if not is_frozen():
            InfoBar.warning(
                title="开发模式",
                content="源码运行时不支持自动更新，请从 GitHub Release 手动下载",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self.window(),
            )
            return

        self._update_btn.setEnabled(False)
        self._update_btn.setText("下载中...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        self._download_worker = DownloadWorker(release_info.download_url, self)
        self._download_worker.progress.connect(self._progress_bar.setValue)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_finished(self, extracted_dir: str):
        self._progress_bar.hide()
        self._update_btn.setText("检查更新")
        self._update_btn.setEnabled(True)

        dlg = MessageBox(
            "下载完成",
            "新版本已下载完成，应用将自动关闭并完成更新后重启。",
            self.window(),
        )
        dlg.yesButton.setText("立即重启")
        dlg.cancelButton.setText("取消")

        if dlg.exec():
            apply_update_and_restart(extracted_dir)

    def _on_download_error(self, msg: str):
        self._progress_bar.hide()
        self._update_btn.setEnabled(True)
        self._update_btn.setText("检查更新")
        InfoBar.error(
            title="下载失败",
            content=msg,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self.window(),
        )

    # ── 加样数据集目录设置 ────────────────────────────────────────

    def _create_sample_dir_card(self, parent_layout: QVBoxLayout) -> None:
        settings = load_settings()
        saved_dir = settings.get("sample_dataset_dir", "")

        card = CardWidget(self)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        icon_widget = IconWidget(FIF.FOLDER, self)
        icon_widget.setFixedSize(32, 32)
        top_row.addWidget(icon_widget)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)
        title_label = BodyLabel("加样数据集存放目录", self)
        desc_label = CaptionLabel(
            "批量标注时标记为"加样"的图片和标注将复制到此目录下的子文件夹中",
            self,
        )
        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)
        top_row.addLayout(text_layout)
        top_row.addStretch()
        card_layout.addLayout(top_row)

        dir_row = QHBoxLayout()
        dir_row.setSpacing(8)

        self._sample_dir_edit = LineEdit(self)
        self._sample_dir_edit.setPlaceholderText("请选择加样数据集存放的根目录")
        self._sample_dir_edit.setText(saved_dir)
        self._sample_dir_edit.editingFinished.connect(self._on_sample_dir_edited)
        dir_row.addWidget(self._sample_dir_edit, 1)

        browse_btn = PushButton("浏览", self)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_sample_dir)
        dir_row.addWidget(browse_btn)

        card_layout.addLayout(dir_row)
        parent_layout.addWidget(card)

    def _on_browse_sample_dir(self):
        current = self._sample_dir_edit.text().strip()
        directory = QFileDialog.getExistingDirectory(
            self, "选择加样数据集存放目录", current or ""
        )
        if directory:
            self._sample_dir_edit.setText(directory)
            self._save_sample_dir_setting()

    def _on_sample_dir_edited(self):
        self._save_sample_dir_setting()

    def _save_sample_dir_setting(self):
        directory = self._sample_dir_edit.text().strip()
        if directory and not os.path.isdir(directory):
            InfoBar.warning(
                title="目录不存在",
                content=f"路径 \"{directory}\" 当前不存在，使用时将尝试自动创建",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self.window(),
            )
        settings = load_settings()
        settings["sample_dataset_dir"] = directory
        save_settings(settings)
