"""
YOLO 训练页面
功能：训练配置、启动/停止、日志显示、权重管理
"""

import io
import os
import re
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QProcess
from PyQt5.QtGui import QFont, QTextCursor
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
)
from qfluentwidgets import (
    PushButton,
    PrimaryPushButton,
    TransparentPushButton,
    CardWidget,
    BodyLabel,
    SubtitleLabel,
    TitleLabel,
    CaptionLabel,
    StrongBodyLabel,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    ComboBox,
    SpinBox,
    ProgressBar,
)

# 从 dataset_page 导入 ProjectManager
from ez_traing.pages.dataset_page import ProjectManager, DatasetProject


def _get_config_dir() -> Path:
    """获取配置目录"""
    config_dir = Path.home() / ".ez_traing"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _detect_devices() -> List[tuple]:
    """
    检测可用的计算设备
    返回: [(device_id, display_name), ...]
    """
    devices = [("cpu", "CPU")]
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                # 获取显存信息
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                display = f"GPU {i}: {name} ({memory_gb:.1f}GB)"
                devices.insert(0, (str(i), display))  # GPU 放在前面
    except ImportError:
        pass
    except Exception:
        pass
    
    return devices


def _strip_ansi(text: str) -> str:
    """去除 ANSI 转义序列"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class YoloTrainThread(QThread):
    """YOLO 训练线程"""
    log_signal = pyqtSignal(str)           # 日志输出
    progress_signal = pyqtSignal(int, int)  # 当前epoch, 总epoch
    finished_signal = pyqtSignal(bool, str) # 成功/失败, 结果路径或错误信息
    
    def __init__(
        self,
        data_yaml: str,
        model: str,
        epochs: int,
        batch_size: int,
        imgsz: int,
        device: str,
        project_dir: str,
        name: str = "train",
    ):
        super().__init__()
        self.data_yaml = data_yaml
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = device
        self.project_dir = project_dir
        self.name = name
        self._stop_requested = False
        self._trainer = None
    
    def run(self):
        try:
            self.log_signal.emit(f"[INFO] 开始加载模型: {self.model}")
            
            # 导入 ultralytics
            from ultralytics import YOLO
            from ultralytics.utils import callbacks
            
            # 加载模型
            model = YOLO(self.model)
            
            # 添加自定义回调来捕获日志
            def on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                epochs = trainer.epochs
                self.progress_signal.emit(epoch, epochs)
                metrics = trainer.metrics
                loss = trainer.loss if hasattr(trainer, 'loss') else 0
                self.log_signal.emit(f"[Epoch {epoch}/{epochs}] loss: {loss:.4f}")
            
            def on_train_batch_end(trainer):
                if self._stop_requested:
                    trainer.stop = True
            
            # 注册回调
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_batch_end", on_train_batch_end)
            
            self.log_signal.emit(f"[INFO] 数据集配置: {self.data_yaml}")
            self.log_signal.emit(f"[INFO] 训练参数: epochs={self.epochs}, batch={self.batch_size}, imgsz={self.imgsz}, device={self.device}")
            self.log_signal.emit(f"[INFO] 输出目录: {self.project_dir}")
            self.log_signal.emit("-" * 50)
            
            # 开始训练
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.imgsz,
                device=self.device if self.device != "auto" else None,
                project=self.project_dir,
                name=self.name,
                exist_ok=True,
                verbose=True,
            )
            
            if self._stop_requested:
                self.log_signal.emit("[INFO] 训练已被用户停止")
                self.finished_signal.emit(False, "训练已停止")
            else:
                save_dir = str(results.save_dir) if hasattr(results, 'save_dir') else self.project_dir
                self.log_signal.emit(f"[INFO] 训练完成！结果保存在: {save_dir}")
                self.finished_signal.emit(True, save_dir)
                
        except Exception as e:
            error_msg = str(e)
            self.log_signal.emit(f"[ERROR] 训练失败: {error_msg}")
            self.finished_signal.emit(False, error_msg)
    
    def stop(self):
        """请求停止训练"""
        self._stop_requested = True
        self.log_signal.emit("[INFO] 正在停止训练...")


class ConfigPanel(CardWidget):
    """训练配置面板"""
    start_training = pyqtSignal(dict)  # 训练配置
    stop_training = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = ProjectManager()
        self._is_training = False
        self._init_ui()
        self._refresh_datasets()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 标题
        title = SubtitleLabel("训练配置", self)
        layout.addWidget(title)
        
        # 数据集选择
        layout.addWidget(StrongBodyLabel("数据集", self))
        self.dataset_combo = ComboBox(self)
        self.dataset_combo.setMinimumWidth(200)
        layout.addWidget(self.dataset_combo)
        
        # 刷新数据集按钮
        refresh_btn = TransparentPushButton("刷新数据集列表", self)
        refresh_btn.clicked.connect(self._refresh_datasets)
        layout.addWidget(refresh_btn)
        
        # 模型选择
        layout.addWidget(StrongBodyLabel("模型", self))
        self.model_combo = ComboBox(self)
        self.model_combo.addItems([
            "yolov8n.pt",
            "yolov8s.pt", 
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ])
        layout.addWidget(self.model_combo)
        
        # 训练参数
        layout.addWidget(StrongBodyLabel("训练参数", self))
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(BodyLabel("Epochs:", self))
        self.epochs_spin = SpinBox(self)
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Batch Size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(BodyLabel("Batch Size:", self))
        self.batch_spin = SpinBox(self)
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        batch_layout.addWidget(self.batch_spin)
        layout.addLayout(batch_layout)
        
        # Image Size
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(BodyLabel("Image Size:", self))
        self.imgsz_spin = SpinBox(self)
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        imgsz_layout.addWidget(self.imgsz_spin)
        layout.addLayout(imgsz_layout)
        
        # Device
        device_layout = QHBoxLayout()
        device_layout.addWidget(BodyLabel("Device:", self))
        self.device_combo = ComboBox(self)
        self._populate_devices()
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)
        
        # 输出目录
        layout.addWidget(StrongBodyLabel("输出目录", self))
        output_layout = QHBoxLayout()
        self.output_dir_label = CaptionLabel("默认: ~/.ez_traing/runs", self)
        self.output_dir_label.setWordWrap(True)
        output_layout.addWidget(self.output_dir_label, 1)
        browse_btn = PushButton("浏览", self)
        browse_btn.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)
        
        self._output_dir = str(_get_config_dir() / "runs")
        
        layout.addStretch()
        
        # 控制按钮
        self.start_btn = PrimaryPushButton("开始训练", self)
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = PushButton("停止训练", self)
        self.stop_btn.setIcon(FIF.PAUSE)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.stop_btn)
    
    def _populate_devices(self):
        """填充可用设备列表"""
        self.device_combo.clear()
        devices = _detect_devices()
        for device_id, display_name in devices:
            self.device_combo.addItem(display_name, device_id)
    
    def _refresh_datasets(self):
        """刷新数据集列表"""
        self._project_manager = ProjectManager()
        self.dataset_combo.clear()
        projects = self._project_manager.get_all_projects()
        for proj in projects:
            self.dataset_combo.addItem(proj.name, proj.id)
    
    def _browse_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self._output_dir
        )
        if dir_path:
            self._output_dir = dir_path
            self.output_dir_label.setText(dir_path)
    
    def _on_start_clicked(self):
        """开始训练"""
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
        
        config = {
            "project": project,
            "model": self.model_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_combo.currentData(),
            "output_dir": self._output_dir,
        }
        
        self.start_training.emit(config)
    
    def _on_stop_clicked(self):
        """停止训练"""
        self.stop_training.emit()
    
    def set_training_state(self, is_training: bool):
        """设置训练状态"""
        self._is_training = is_training
        self.start_btn.setEnabled(not is_training)
        self.stop_btn.setEnabled(is_training)
        self.dataset_combo.setEnabled(not is_training)
        self.model_combo.setEnabled(not is_training)
        self.epochs_spin.setEnabled(not is_training)
        self.batch_spin.setEnabled(not is_training)
        self.imgsz_spin.setEnabled(not is_training)
        self.device_combo.setEnabled(not is_training)


class LogPanel(CardWidget):
    """日志显示面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # 标题栏
        header_layout = QHBoxLayout()
        title = SubtitleLabel("训练日志", self)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        clear_btn = TransparentPushButton("清空", self)
        clear_btn.clicked.connect(self._clear_log)
        header_layout.addWidget(clear_btn)
        layout.addLayout(header_layout)
        
        # 进度条
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = CaptionLabel("等待训练开始...", self)
        layout.addWidget(self.progress_label)
        
        # 日志文本框
        self.log_text = QPlainTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.log_text)
    
    def append_log(self, text: str):
        """追加日志"""
        text = _strip_ansi(text)
        self.log_text.appendPlainText(text)
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
    
    def set_progress(self, current: int, total: int):
        """设置进度"""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Epoch {current}/{total} ({percent}%)")
    
    def reset(self):
        """重置状态"""
        self.progress_bar.setValue(0)
        self.progress_label.setText("等待训练开始...")
    
    def _clear_log(self):
        """清空日志"""
        self.log_text.clear()


class WeightPanel(CardWidget):
    """权重管理面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._runs_dir = str(_get_config_dir() / "runs")
        self._init_ui()
        self._refresh_weights()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # 标题栏
        header_layout = QHBoxLayout()
        title = SubtitleLabel("权重管理", self)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        refresh_btn = TransparentPushButton("刷新", self)
        refresh_btn.clicked.connect(self._refresh_weights)
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)
        
        # 训练结果目录列表
        layout.addWidget(StrongBodyLabel("训练记录", self))
        self.run_list = QListWidget(self)
        self.run_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.run_list.itemSelectionChanged.connect(self._on_run_selected)
        self.run_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        layout.addWidget(self.run_list)
        
        # 权重文件列表
        layout.addWidget(StrongBodyLabel("权重文件", self))
        self.weight_list = QListWidget(self)
        self.weight_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.weight_list.setStyleSheet(self.run_list.styleSheet())
        layout.addWidget(self.weight_list)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        self.open_dir_btn = PushButton("打开目录", self)
        self.open_dir_btn.setIcon(FIF.FOLDER)
        self.open_dir_btn.clicked.connect(self._open_selected_dir)
        btn_layout.addWidget(self.open_dir_btn)
        layout.addLayout(btn_layout)
    
    def _refresh_weights(self):
        """刷新权重列表"""
        self.run_list.clear()
        self.weight_list.clear()
        
        runs_path = Path(self._runs_dir)
        if not runs_path.exists():
            return
        
        # 查找所有 detect 目录下的训练记录
        detect_path = runs_path / "detect"
        if detect_path.exists():
            for run_dir in sorted(detect_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if run_dir.is_dir():
                    # 检查是否有权重文件
                    weights_dir = run_dir / "weights"
                    if weights_dir.exists():
                        item = QListWidgetItem(run_dir.name)
                        item.setData(Qt.UserRole, str(run_dir))
                        self.run_list.addItem(item)
    
    def _on_run_selected(self):
        """选中训练记录时更新权重列表"""
        self.weight_list.clear()
        
        selected = self.run_list.selectedItems()
        if not selected:
            return
        
        run_dir = Path(selected[0].data(Qt.UserRole))
        weights_dir = run_dir / "weights"
        
        if weights_dir.exists():
            for weight_file in sorted(weights_dir.glob("*.pt")):
                item = QListWidgetItem(weight_file.name)
                item.setData(Qt.UserRole, str(weight_file))
                # 显示文件大小
                size_mb = weight_file.stat().st_size / (1024 * 1024)
                item.setToolTip(f"大小: {size_mb:.1f} MB")
                self.weight_list.addItem(item)
    
    def _open_selected_dir(self):
        """打开选中的目录"""
        selected = self.run_list.selectedItems()
        if selected:
            run_dir = selected[0].data(Qt.UserRole)
            os.startfile(run_dir)
        else:
            # 打开 runs 目录
            runs_path = Path(self._runs_dir)
            if runs_path.exists():
                os.startfile(str(runs_path))
    
    def set_runs_dir(self, dir_path: str):
        """设置 runs 目录"""
        self._runs_dir = dir_path
        self._refresh_weights()


class TrainPage(QWidget):
    """YOLO 训练页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._train_thread: Optional[YoloTrainThread] = None
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题栏
        header = CardWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        title = SubtitleLabel("YOLO 训练", self)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        self.status_label = CaptionLabel("就绪", self)
        header_layout.addWidget(self.status_label)
        
        header.setFixedHeight(40)
        main_layout.addWidget(header)
        
        # 内容区域 - 三栏布局
        content_splitter = QSplitter(Qt.Horizontal, self)
        
        # 左侧：配置面板
        self.config_panel = ConfigPanel(self)
        self.config_panel.setMinimumWidth(260)
        self.config_panel.setMaximumWidth(320)
        content_splitter.addWidget(self.config_panel)
        
        # 中间：日志面板
        self.log_panel = LogPanel(self)
        content_splitter.addWidget(self.log_panel)
        
        # 右侧：权重管理面板
        self.weight_panel = WeightPanel(self)
        self.weight_panel.setMinimumWidth(260)
        self.weight_panel.setMaximumWidth(350)
        content_splitter.addWidget(self.weight_panel)
        
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setStretchFactor(2, 0)
        
        main_layout.addWidget(content_splitter)
    
    def _connect_signals(self):
        """连接信号"""
        self.config_panel.start_training.connect(self._start_training)
        self.config_panel.stop_training.connect(self._stop_training)
    
    def _generate_data_yaml(self, project: DatasetProject, output_dir: str) -> str:
        """生成数据集 YAML 配置文件"""
        dataset_dir = Path(project.directory)
        
        # 查找 classes.txt
        classes_file = None
        for path in [
            dataset_dir / "classes.txt",
            dataset_dir / "labels" / "classes.txt",
        ]:
            if path.exists():
                classes_file = path
                break
        
        if not classes_file:
            raise ValueError(f"找不到 classes.txt 文件: {dataset_dir}")
        
        # 读取类别
        with open(classes_file, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        
        if not class_names:
            raise ValueError("classes.txt 为空")
        
        # 确定图片目录结构
        # 支持多种常见结构:
        # 1. dataset/images + dataset/labels
        # 2. dataset/images/train + dataset/images/val
        # 3. 直接在 dataset 目录下
        
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            images_dir = dataset_dir
        
        train_dir = images_dir / "train"
        val_dir = images_dir / "val"
        
        if train_dir.exists() and val_dir.exists():
            train_path = str(train_dir)
            val_path = str(val_dir)
        else:
            # 使用同一目录作为训练和验证集
            train_path = str(images_dir)
            val_path = str(images_dir)
        
        # 生成 YAML 配置
        data_config = {
            "path": str(dataset_dir),
            "train": train_path,
            "val": val_path,
            "names": {i: name for i, name in enumerate(class_names)},
        }
        
        # 保存 YAML 文件
        yaml_path = Path(output_dir) / f"{project.name}_data.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
        
        return str(yaml_path)
    
    def _start_training(self, config: dict):
        """开始训练"""
        try:
            project: DatasetProject = config["project"]
            
            # 生成数据集配置文件
            data_yaml = self._generate_data_yaml(project, config["output_dir"])
            
            self.log_panel.reset()
            self.log_panel.append_log(f"[INFO] 数据集: {project.name}")
            self.log_panel.append_log(f"[INFO] 目录: {project.directory}")
            self.log_panel.append_log(f"[INFO] 生成配置文件: {data_yaml}")
            
            # 创建训练线程
            self._train_thread = YoloTrainThread(
                data_yaml=data_yaml,
                model=config["model"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                imgsz=config["imgsz"],
                device=config["device"],
                project_dir=config["output_dir"],
                name=project.name,
            )
            
            # 连接信号
            self._train_thread.log_signal.connect(self.log_panel.append_log)
            self._train_thread.progress_signal.connect(self.log_panel.set_progress)
            self._train_thread.finished_signal.connect(self._on_training_finished)
            
            # 更新 UI 状态
            self.config_panel.set_training_state(True)
            self.status_label.setText("训练中...")
            
            # 启动训练
            self._train_thread.start()
            
        except Exception as e:
            InfoBar.error(
                title="启动失败",
                content=str(e),
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
    
    def _stop_training(self):
        """停止训练"""
        if self._train_thread and self._train_thread.isRunning():
            self._train_thread.stop()
            self.status_label.setText("正在停止...")
    
    def _on_training_finished(self, success: bool, result: str):
        """训练完成回调"""
        self.config_panel.set_training_state(False)
        
        if success:
            self.status_label.setText("训练完成")
            self.weight_panel._refresh_weights()
            InfoBar.success(
                title="训练完成",
                content=f"结果保存在: {result}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            self.status_label.setText("训练失败")
            InfoBar.error(
                title="训练失败",
                content=result,
                parent=self.window(),
                position=InfoBarPosition.TOP,
            )
        
        self._train_thread = None
