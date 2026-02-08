"""设置页面 - 显示环境信息和 GPU 状态"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout
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
)
from PyQt5.QtGui import QClipboard
from PyQt5.QtWidgets import QApplication


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
