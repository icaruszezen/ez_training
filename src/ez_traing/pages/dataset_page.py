"""
数据集管理页面
功能：项目管理、导入、目录扫描、数据预览、标注联动
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QImage, QColor, QPainter, QBrush, QPen
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QFileDialog,
    QSplitter,
    QFrame,
    QAbstractItemView,
    QInputDialog,
    QMessageBox,
    QScrollArea,
    QGridLayout,
    QComboBox,
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
    ProgressBar,
)

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS

# 项目配置文件路径
def _get_config_dir() -> Path:
    """获取配置目录"""
    config_dir = Path.home() / ".ez_traing"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def _get_projects_file() -> Path:
    """获取项目配置文件路径"""
    return _get_config_dir() / "datasets.json"


@dataclass
class DatasetProject:
    """数据集项目"""
    id: str
    name: str
    directory: str
    image_count: int = 0
    annotated_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    
    @classmethod
    def create(cls, name: str, directory: str) -> "DatasetProject":
        """创建新项目"""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            directory=directory,
            created_at=now,
            updated_at=now,
        )


class ProjectManager:
    """项目管理器"""
    
    def __init__(self):
        self.projects: Dict[str, DatasetProject] = {}
        self._load()
    
    def _load(self):
        """加载项目配置"""
        config_file = _get_projects_file()
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("projects", []):
                        proj = DatasetProject(**item)
                        self.projects[proj.id] = proj
            except Exception:
                pass
    
    def _save(self):
        """保存项目配置"""
        config_file = _get_projects_file()
        data = {
            "projects": [asdict(p) for p in self.projects.values()]
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_project(self, project: DatasetProject):
        """添加项目"""
        self.projects[project.id] = project
        self._save()
    
    def remove_project(self, project_id: str):
        """删除项目"""
        if project_id in self.projects:
            del self.projects[project_id]
            self._save()
    
    def update_project(self, project: DatasetProject):
        """更新项目"""
        project.updated_at = datetime.now().isoformat()
        self.projects[project.id] = project
        self._save()
    
    def get_project(self, project_id: str) -> Optional[DatasetProject]:
        """获取项目"""
        return self.projects.get(project_id)
    
    def get_all_projects(self) -> List[DatasetProject]:
        """获取所有项目"""
        return list(self.projects.values())


@dataclass
class AnnotationStats:
    """标注统计信息"""
    total_images: int = 0
    annotated_images: int = 0
    unannotated_images: int = 0
    total_objects: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def annotation_rate(self) -> float:
        """标注率"""
        if self.total_images == 0:
            return 0.0
        return self.annotated_images / self.total_images * 100


@dataclass
class ImageInfo:
    """图片元信息（用于筛选）"""
    path: str
    is_annotated: bool
    image_type: str  # 小写扩展名，不含点，例如 jpg / png


class ImageScanner(QThread):
    """异步图片扫描线程"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list, object)  # image_infos, AnnotationStats
    
    def __init__(self, directory: str, classes_file: str = None):
        super().__init__()
        self.directory = directory
        self.classes_file = classes_file
        self._is_cancelled = False
        self._class_names = []  # YOLO 类别名称列表
        self._voc_label_cache: Dict[tuple[str, int], List[str]] = {}
    
    def _load_classes(self):
        """加载 YOLO classes.txt 文件"""
        # 尝试多个可能的位置
        possible_paths = [
            Path(self.directory) / "classes.txt",
            Path(self.directory) / "labels" / "classes.txt",
            Path(self.directory) / ".." / "classes.txt",
        ]
        if self.classes_file:
            possible_paths.insert(0, Path(self.classes_file))
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self._class_names = [line.strip() for line in f if line.strip()]
                    return
                except Exception:
                    pass
    
    def _parse_yolo_annotation(self, txt_path: Path) -> List[str]:
        """解析 YOLO 格式标注文件，返回标签列表"""
        labels = []
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO format: class_id x y w h
                        try:
                            class_id = int(parts[0])
                            if self._class_names and 0 <= class_id < len(self._class_names):
                                labels.append(self._class_names[class_id])
                            else:
                                labels.append(f"class_{class_id}")
                        except ValueError:
                            pass
        except Exception:
            pass
        return labels
    
    def _parse_voc_annotation(self, xml_path: Path) -> List[str]:
        """解析 VOC 格式标注文件，返回标签列表"""
        labels = []
        try:
            resolved = xml_path.resolve()
            mtime_ns = resolved.stat().st_mtime_ns
            cache_key = (str(resolved), mtime_ns)
            cached = self._voc_label_cache.get(cache_key)
            if cached is not None:
                return list(cached)
        except OSError:
            resolved = xml_path
            cache_key = None
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(resolved)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name")
                if name is not None and name.text:
                    labels.append(name.text)
        except Exception:
            pass
        if cache_key is not None:
            stale_keys = [k for k in self._voc_label_cache.keys() if k[0] == cache_key[0] and k != cache_key]
            for key in stale_keys:
                self._voc_label_cache.pop(key, None)
            self._voc_label_cache[cache_key] = list(labels)
        return labels
    
    def run(self):
        image_infos = []
        all_files = []
        stats = AnnotationStats()
        label_counter = Counter()
        
        # 加载类别名称
        self._load_classes()
        
        # 收集所有文件
        for root, _, files in os.walk(self.directory):
            for file in files:
                if Path(file).suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                    all_files.append(os.path.join(root, file))
        
        total = len(all_files)
        stats.total_images = total
        
        for i, file_path in enumerate(all_files):
            if self._is_cancelled:
                break
            
            path = Path(file_path)
            labels = []
            
            # 检查 YOLO 格式标注 (.txt)
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                labels = self._parse_yolo_annotation(txt_path)
            else:
                # 检查 VOC 格式标注 (.xml)
                xml_path = path.with_suffix(".xml")
                if xml_path.exists():
                    labels = self._parse_voc_annotation(xml_path)
            
            is_annotated = bool(labels)
            if is_annotated:
                stats.annotated_images += 1
                stats.total_objects += len(labels)
                label_counter.update(labels)

            image_infos.append(
                ImageInfo(
                    path=file_path,
                    is_annotated=is_annotated,
                    image_type=path.suffix.lower().lstrip("."),
                )
            )
            
            self.progress.emit(i + 1, total)
        
        stats.unannotated_images = stats.total_images - stats.annotated_images
        stats.label_counts = dict(label_counter)
        
        self.finished.emit(image_infos, stats)
    
    def cancel(self):
        self._is_cancelled = True


class ThumbnailLoader(QThread):
    """异步缩略图加载线程 - 使用 QImage 避免线程安全问题"""
    thumbnail_loaded = pyqtSignal(str, QImage)  # path, image (QImage 可跨线程)
    all_loaded = pyqtSignal()
    
    def __init__(self, image_paths: List[str], thumbnail_size: int = 120):
        super().__init__()
        self.image_paths = image_paths
        self.thumbnail_size = thumbnail_size
        self._is_cancelled = False
    
    def run(self):
        for path in self.image_paths:
            if self._is_cancelled:
                break
            try:
                # 使用 QImage 而非 QPixmap（QImage 是线程安全的）
                image = QImage(path)
                if not image.isNull():
                    scaled = image.scaled(
                        self.thumbnail_size, 
                        self.thumbnail_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.thumbnail_loaded.emit(path, scaled)
            except Exception:
                pass
        self.all_loaded.emit()
    
    def cancel(self):
        self._is_cancelled = True


class ProjectListWidget(CardWidget):
    """项目列表组件"""
    
    project_selected = pyqtSignal(str)  # project_id
    project_deleted = pyqtSignal(str)   # project_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # 标题
        title_label = SubtitleLabel("数据集项目")
        layout.addWidget(title_label)
        
        # 项目列表
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                outline: none;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #eeeeee;
            }
            QListWidget::item:last-child {
                border-bottom: none;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        self.project_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.project_list, 1)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.add_btn = PrimaryPushButton("新建项目", self, FIF.ADD)
        self.add_btn.setFixedHeight(36)
        btn_layout.addWidget(self.add_btn)
        
        self.delete_btn = TransparentPushButton("删除", self, FIF.DELETE)
        self.delete_btn.setFixedHeight(36)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)
        
        layout.addLayout(btn_layout)
    
    def add_project_item(self, project: DatasetProject, select: bool = False):
        """添加项目项"""
        item = QListWidgetItem()
        item.setData(Qt.UserRole, project.id)
        
        # 格式化显示文本
        text = f"{project.name}\n"
        text += f"📁 {project.directory}\n"
        text += f"🖼 {project.image_count} 张图片 · ✅ {project.annotated_count} 已标注"
        item.setText(text)
        item.setSizeHint(QSize(0, 80))
        
        self.project_list.addItem(item)
        
        if select:
            self.project_list.setCurrentItem(item)
            self.delete_btn.setEnabled(True)
    
    def update_project_item(self, project: DatasetProject):
        """更新项目项"""
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.data(Qt.UserRole) == project.id:
                text = f"{project.name}\n"
                text += f"📁 {project.directory}\n"
                text += f"🖼 {project.image_count} 张图片 · ✅ {project.annotated_count} 已标注"
                item.setText(text)
                break
    
    def remove_project_item(self, project_id: str):
        """移除项目项"""
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.data(Qt.UserRole) == project_id:
                self.project_list.takeItem(i)
                break
        
        if self.project_list.count() == 0:
            self.delete_btn.setEnabled(False)
    
    def clear_projects(self):
        """清空项目列表"""
        self.project_list.clear()
        self.delete_btn.setEnabled(False)
    
    def get_selected_project_id(self) -> Optional[str]:
        """获取选中的项目ID"""
        items = self.project_list.selectedItems()
        if items:
            return items[0].data(Qt.UserRole)
        return None
    
    def _on_item_clicked(self, item: QListWidgetItem):
        """项目点击"""
        project_id = item.data(Qt.UserRole)
        self.delete_btn.setEnabled(True)
        self.project_selected.emit(project_id)


# 预设颜色列表（用于标签显示）
LABEL_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FFEB3B", "#795548", "#607D8B", "#F44336",
    "#3F51B5", "#8BC34A", "#FF5722", "#673AB7", "#009688",
]


class StatisticsPanel(CardWidget):
    """标注统计面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._stats: Optional[AnnotationStats] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 标题
        title_label = SubtitleLabel("标注统计")
        layout.addWidget(title_label)
        
        # 概览卡片
        overview_card = QFrame()
        overview_card.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        overview_layout = QGridLayout(overview_card)
        overview_layout.setContentsMargins(12, 12, 12, 12)
        overview_layout.setSpacing(8)
        
        # 总图片数
        self.total_label = StrongBodyLabel("0")
        self.total_label.setAlignment(Qt.AlignCenter)
        overview_layout.addWidget(CaptionLabel("总图片"), 0, 0, Qt.AlignCenter)
        overview_layout.addWidget(self.total_label, 1, 0, Qt.AlignCenter)
        
        # 已标注
        self.annotated_label = StrongBodyLabel("0")
        self.annotated_label.setAlignment(Qt.AlignCenter)
        self.annotated_label.setStyleSheet("color: #4CAF50;")
        overview_layout.addWidget(CaptionLabel("已标注"), 0, 1, Qt.AlignCenter)
        overview_layout.addWidget(self.annotated_label, 1, 1, Qt.AlignCenter)
        
        # 未标注
        self.unannotated_label = StrongBodyLabel("0")
        self.unannotated_label.setAlignment(Qt.AlignCenter)
        self.unannotated_label.setStyleSheet("color: #FF9800;")
        overview_layout.addWidget(CaptionLabel("未标注"), 0, 2, Qt.AlignCenter)
        overview_layout.addWidget(self.unannotated_label, 1, 2, Qt.AlignCenter)
        
        layout.addWidget(overview_card)
        
        # 标注率
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(CaptionLabel("标注进度:"))
        self.rate_label = BodyLabel("0%")
        rate_layout.addWidget(self.rate_label)
        rate_layout.addStretch()
        layout.addLayout(rate_layout)
        
        # 进度条
        self.rate_bar = ProgressBar()
        self.rate_bar.setValue(0)
        layout.addWidget(self.rate_bar)
        
        # 对象总数
        objects_layout = QHBoxLayout()
        objects_layout.addWidget(CaptionLabel("标注对象总数:"))
        self.objects_label = BodyLabel("0")
        objects_layout.addWidget(self.objects_label)
        objects_layout.addStretch()
        layout.addLayout(objects_layout)
        
        # 标签分布标题
        labels_header = QHBoxLayout()
        labels_header.addWidget(CaptionLabel("标签分布"))
        labels_header.addStretch()
        self.labels_count_label = CaptionLabel("0 种标签")
        labels_header.addWidget(self.labels_count_label)
        layout.addLayout(labels_header)
        
        # 标签列表（滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
            }
        """)
        
        self.labels_container = QWidget()
        self.labels_layout = QVBoxLayout(self.labels_container)
        self.labels_layout.setContentsMargins(8, 8, 8, 8)
        self.labels_layout.setSpacing(6)
        self.labels_layout.addStretch()
        
        scroll_area.setWidget(self.labels_container)
        layout.addWidget(scroll_area, 1)
    
    def set_stats(self, stats: AnnotationStats):
        """设置统计数据"""
        self._stats = stats
        
        # 更新概览
        self.total_label.setText(str(stats.total_images))
        self.annotated_label.setText(str(stats.annotated_images))
        self.unannotated_label.setText(str(stats.unannotated_images))
        
        # 更新标注率
        rate = stats.annotation_rate
        self.rate_label.setText(f"{rate:.1f}%")
        self.rate_bar.setValue(int(rate))
        
        # 更新对象总数
        self.objects_label.setText(str(stats.total_objects))
        
        # 更新标签分布
        self._update_labels(stats.label_counts)
    
    def _update_labels(self, label_counts: Dict[str, int]):
        """更新标签分布显示"""
        # 清空现有标签
        while self.labels_layout.count() > 1:  # 保留 stretch
            item = self.labels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not label_counts:
            self.labels_count_label.setText("0 种标签")
            no_label = CaptionLabel("暂无标签数据")
            no_label.setStyleSheet("color: #999999;")
            self.labels_layout.insertWidget(0, no_label)
            return
        
        self.labels_count_label.setText(f"{len(label_counts)} 种标签")
        
        # 计算总数用于百分比
        total = sum(label_counts.values())
        
        # 按数量排序
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (label, count) in enumerate(sorted_labels):
            color = LABEL_COLORS[i % len(LABEL_COLORS)]
            percentage = count / total * 100 if total > 0 else 0
            
            # 创建标签行
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            
            # 颜色指示器
            color_dot = QLabel()
            color_dot.setFixedSize(12, 12)
            color_dot.setStyleSheet(f"""
                background-color: {color};
                border-radius: 6px;
            """)
            row_layout.addWidget(color_dot)
            
            # 标签名
            name_label = BodyLabel(label)
            name_label.setStyleSheet("color: #333333;")
            row_layout.addWidget(name_label, 1)
            
            # 数量和百分比
            count_label = CaptionLabel(f"{count} ({percentage:.1f}%)")
            count_label.setStyleSheet("color: #666666;")
            row_layout.addWidget(count_label)
            
            self.labels_layout.insertWidget(self.labels_layout.count() - 1, row)
    
    def clear(self):
        """清空统计"""
        self._stats = None
        self.total_label.setText("0")
        self.annotated_label.setText("0")
        self.unannotated_label.setText("0")
        self.rate_label.setText("0%")
        self.rate_bar.setValue(0)
        self.objects_label.setText("0")
        self._update_labels({})


class ImagePreviewWidget(QFrame):
    """图片预览组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(280)
        self._setup_ui()
        self._current_path: Optional[str] = None
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 标题
        self.title_label = SubtitleLabel("图片预览")
        layout.addWidget(self.title_label)
        
        # 图片预览区域
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(200, 200)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        self.preview_label.setText("选择图片以预览")
        layout.addWidget(self.preview_label, 1)
        
        # 图片信息
        info_card = CardWidget()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(6)
        
        self.file_name_label = BodyLabel("文件名: -")
        self.file_size_label = CaptionLabel("大小: -")
        self.image_size_label = CaptionLabel("尺寸: -")
        self.annotation_status_label = CaptionLabel("标注: -")
        
        info_layout.addWidget(self.file_name_label)
        info_layout.addWidget(self.file_size_label)
        info_layout.addWidget(self.image_size_label)
        info_layout.addWidget(self.annotation_status_label)
        
        layout.addWidget(info_card)
    
    def set_image(self, image_path: str):
        """设置预览图片"""
        self._current_path = image_path
        
        if not image_path or not os.path.exists(image_path):
            self.preview_label.setText("选择图片以预览")
            self._clear_info()
            return
        
        # 加载图片
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.preview_label.setText("无法加载图片")
            self._clear_info()
            return
        
        # 缩放显示
        preview_size = self.preview_label.size()
        scaled = pixmap.scaled(
            preview_size.width() - 20,
            preview_size.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)
        
        # 更新信息
        file_path = Path(image_path)
        file_stat = file_path.stat()
        
        self.file_name_label.setText(f"文件名: {file_path.name}")
        self.file_size_label.setText(f"大小: {self._format_size(file_stat.st_size)}")
        self.image_size_label.setText(f"尺寸: {pixmap.width()} × {pixmap.height()}")
        
        # 检查标注状态
        annotation_status = self._check_annotation_status(image_path)
        self.annotation_status_label.setText(f"标注: {annotation_status}")
    
    def _clear_info(self):
        self.file_name_label.setText("文件名: -")
        self.file_size_label.setText("大小: -")
        self.image_size_label.setText("尺寸: -")
        self.annotation_status_label.setText("标注: -")
    
    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _check_annotation_status(self, image_path: str) -> str:
        """检查图片的标注状态"""
        path = Path(image_path)
        
        # 检查 YOLO 格式标注 (.txt)
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r") as f:
                lines = [l for l in f.readlines() if l.strip()]
                if lines:
                    return f"已标注 (YOLO, {len(lines)} 个对象)"
        
        # 检查 VOC 格式标注 (.xml)
        xml_path = path.with_suffix(".xml")
        if xml_path.exists():
            return "已标注 (VOC)"
        
        return "未标注"
    
    @property
    def current_path(self) -> Optional[str]:
        return self._current_path


class ImageListPanel(CardWidget):
    """图片列表面板"""
    
    image_selected = pyqtSignal(str)  # image_path
    image_double_clicked = pyqtSignal(str)  # image_path
    filters_changed = pyqtSignal(str, str)  # annotation_filter, type_filter
    
    BATCH_SIZE = 50  # 每批添加的图片数量
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._thumbnail_cache = {}
        self._path_to_item = {}  # 路径到 item 的映射，加速查找
        self._pending_paths = []  # 待添加的路径
        self._placeholder_pixmap = None
        self._add_timer = QTimer(self)
        self._add_timer.timeout.connect(self._add_batch)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # 标题和统计
        header_layout = QHBoxLayout()
        self.title_label = SubtitleLabel("图片列表")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.count_label = CaptionLabel("共 0 张图片")
        header_layout.addWidget(self.count_label)
        layout.addLayout(header_layout)

        # 筛选栏
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)
        filter_layout.addWidget(CaptionLabel("标注:"))

        self.annotation_filter = QComboBox()
        self.annotation_filter.addItems(["全部", "已标注", "未标注"])
        self.annotation_filter.currentIndexChanged.connect(self._on_filters_changed)
        filter_layout.addWidget(self.annotation_filter)

        filter_layout.addWidget(CaptionLabel("类型:"))
        self.type_filter = QComboBox()
        self.type_filter.addItem("全部")
        self.type_filter.currentIndexChanged.connect(self._on_filters_changed)
        filter_layout.addWidget(self.type_filter)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # 图片列表
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.IconMode)
        self.image_list.setIconSize(QSize(120, 120))
        self.image_list.setSpacing(10)
        self.image_list.setResizeMode(QListWidget.Adjust)
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.setMovement(QListWidget.Static)
        self.image_list.setUniformItemSizes(True)  # 优化：统一项目大小
        self.image_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 6px;
                color: #333333;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                border: 2px solid #2196f3;
                color: #1565c0;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        self.image_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.image_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.image_list, 1)
    
    def set_images(self, image_paths: List[str], reset_cache: bool = False):
        """设置图片列表 - 分批添加避免卡顿"""
        self._add_timer.stop()
        self.image_list.clear()
        self._path_to_item.clear()
        if reset_cache:
            self._thumbnail_cache.clear()
        
        count = len(image_paths)
        self.count_label.setText(f"共 {count} 张图片")
        
        if count == 0:
            return
        
        # 准备占位图
        if self._placeholder_pixmap is None:
            self._placeholder_pixmap = self._create_placeholder_pixmap()
        
        # 分批添加
        self._pending_paths = list(image_paths)
        self._add_timer.start(10)  # 每 10ms 添加一批
    
    def _add_batch(self):
        """添加一批图片项"""
        if not self._pending_paths:
            self._add_timer.stop()
            return
        
        # 取出一批
        batch = self._pending_paths[:self.BATCH_SIZE]
        self._pending_paths = self._pending_paths[self.BATCH_SIZE:]
        
        # 批量添加
        for path in batch:
            item = QListWidgetItem()
            cached = self._thumbnail_cache.get(path)
            if cached:
                item.setIcon(QIcon(cached))
            else:
                item.setIcon(QIcon(self._placeholder_pixmap))
            item.setText(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setSizeHint(QSize(140, 160))
            self.image_list.addItem(item)
            self._path_to_item[path] = item
        
        # 如果添加完成，停止定时器
        if not self._pending_paths:
            self._add_timer.stop()
    
    def update_thumbnail(self, path: str, pixmap: QPixmap):
        """更新缩略图 - 使用字典快速查找"""
        self._thumbnail_cache[path] = pixmap
        
        item = self._path_to_item.get(path)
        if item:
            item.setIcon(QIcon(pixmap))
    
    def clear(self):
        """清空列表"""
        self._add_timer.stop()
        self._pending_paths.clear()
        self.image_list.clear()
        self._thumbnail_cache.clear()
        self._path_to_item.clear()
        self.count_label.setText("共 0 张图片")
        self.reset_filters()
        self.set_type_options([])

    def set_type_options(self, image_types: List[str]):
        """设置类型筛选选项"""
        current = self.type_filter.currentText()
        self.type_filter.blockSignals(True)
        self.type_filter.clear()
        self.type_filter.addItem("全部")
        for image_type in image_types:
            self.type_filter.addItem(image_type.upper())
        if current in ["全部", ""] or current not in [t.upper() for t in image_types]:
            self.type_filter.setCurrentIndex(0)
        else:
            self.type_filter.setCurrentText(current)
        self.type_filter.blockSignals(False)

    def reset_filters(self):
        """重置筛选"""
        self.annotation_filter.blockSignals(True)
        self.type_filter.blockSignals(True)
        self.annotation_filter.setCurrentIndex(0)
        self.type_filter.setCurrentIndex(0)
        self.annotation_filter.blockSignals(False)
        self.type_filter.blockSignals(False)

    def _on_filters_changed(self):
        self.filters_changed.emit(
            self.annotation_filter.currentText(),
            self.type_filter.currentText(),
        )
    
    def _create_placeholder_pixmap(self) -> QPixmap:
        """创建占位图"""
        pixmap = QPixmap(120, 120)
        pixmap.fill(Qt.lightGray)
        return pixmap
    
    def _on_selection_changed(self):
        """选择变化"""
        items = self.image_list.selectedItems()
        if items:
            path = items[0].data(Qt.UserRole)
            self.image_selected.emit(path)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """双击项目"""
        path = item.data(Qt.UserRole)
        if path:
            self.image_double_clicked.emit(path)
    
    def get_selected_path(self) -> Optional[str]:
        """获取选中的图片路径"""
        items = self.image_list.selectedItems()
        if items:
            return items[0].data(Qt.UserRole)
        return None


class DatasetPage(QWidget):
    """数据集管理页面"""
    
    # 信号：请求打开图片进行标注 (目录路径, 图片路径)
    request_annotation = pyqtSignal(str, str)  # directory, image_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = ProjectManager()
        self.current_project: Optional[DatasetProject] = None
        self.image_infos: List[ImageInfo] = []
        self.image_paths: List[str] = []
        self.filtered_image_paths: List[str] = []
        self._scanner: Optional[ImageScanner] = None
        self._thumbnail_loader: Optional[ThumbnailLoader] = None
        
        self._setup_ui()
        self._load_projects()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 顶部标题和操作栏
        header = self._create_header()
        main_layout.addWidget(header)
        
        # 进度条
        self.progress_bar = ProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 主内容区域
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：项目列表
        self.project_list_widget = ProjectListWidget()
        self.project_list_widget.project_selected.connect(self._on_project_selected)
        self.project_list_widget.add_btn.clicked.connect(self._on_add_project)
        self.project_list_widget.delete_btn.clicked.connect(self._on_delete_project)
        content_splitter.addWidget(self.project_list_widget)
        
        # 中间：图片列表
        self.image_list_panel = ImageListPanel()
        self.image_list_panel.image_selected.connect(self._on_image_selected)
        self.image_list_panel.image_double_clicked.connect(self._on_image_double_clicked)
        self.image_list_panel.filters_changed.connect(self._on_filters_changed)
        content_splitter.addWidget(self.image_list_panel)
        
        # 右侧：统计和预览（垂直分割）
        right_splitter = QSplitter(Qt.Vertical)
        
        # 统计面板
        self.statistics_panel = StatisticsPanel()
        right_splitter.addWidget(self.statistics_panel)
        
        # 预览区域
        self.preview_widget = ImagePreviewWidget()
        right_splitter.addWidget(self.preview_widget)
        
        right_splitter.setSizes([350, 350])
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        
        content_splitter.addWidget(right_splitter)
        
        # 设置分割比例
        content_splitter.setSizes([260, 500, 340])
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setStretchFactor(2, 1)
        
        main_layout.addWidget(content_splitter, 1)
        
        # 底部状态栏
        self.status_label = CaptionLabel("选择或创建数据集项目")
        main_layout.addWidget(self.status_label)
    
    def _create_header(self) -> QWidget:
        """创建顶部标题栏"""
        header = CardWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(16)
        
        # 标题
        title = TitleLabel("数据集管理")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # 刷新按钮
        self.refresh_btn = PushButton("刷新", self, FIF.SYNC)
        self.refresh_btn.clicked.connect(self._on_refresh_project)
        self.refresh_btn.setEnabled(False)
        layout.addWidget(self.refresh_btn)
        
        # 打开标注按钮
        self.annotate_btn = PrimaryPushButton("打开标注", self, FIF.EDIT)
        self.annotate_btn.setEnabled(False)
        self.annotate_btn.clicked.connect(self._on_annotate_clicked)
        layout.addWidget(self.annotate_btn)
        
        return header
    
    def _load_projects(self):
        """加载所有项目"""
        self.project_list_widget.clear_projects()
        projects = self.project_manager.get_all_projects()
        for project in projects:
            self.project_list_widget.add_project_item(project)
    
    def _on_add_project(self):
        """添加新项目"""
        # 选择目录
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择数据集目录",
            "",
            QFileDialog.ShowDirsOnly
        )
        if not directory:
            return
        
        # 输入项目名称
        default_name = Path(directory).name
        name, ok = QInputDialog.getText(
            self,
            "新建数据集项目",
            "项目名称:",
            text=default_name
        )
        if not ok or not name.strip():
            return
        
        # 创建项目
        project = DatasetProject.create(name.strip(), directory)
        self.project_manager.add_project(project)
        self.project_list_widget.add_project_item(project, select=True)
        
        # 自动扫描
        self._load_project(project)
        
        InfoBar.success(
            title="成功",
            content=f"已创建项目: {name}",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def _on_delete_project(self):
        """删除项目"""
        project_id = self.project_list_widget.get_selected_project_id()
        if not project_id:
            return
        
        project = self.project_manager.get_project(project_id)
        if not project:
            return
        
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除项目 \"{project.name}\" 吗？\n\n（只删除项目记录，不会删除实际文件）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.project_manager.remove_project(project_id)
            self.project_list_widget.remove_project_item(project_id)
            
            if self.current_project and self.current_project.id == project_id:
                self.current_project = None
                self.image_infos.clear()
                self.image_paths.clear()
                self.filtered_image_paths.clear()
                self.image_list_panel.clear()
                self.preview_widget.set_image(None)
                self.statistics_panel.clear()
                self.refresh_btn.setEnabled(False)
                self.annotate_btn.setEnabled(False)
                self.status_label.setText("选择或创建数据集项目")
            
            InfoBar.success(
                title="成功",
                content=f"已删除项目: {project.name}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
    
    def _on_project_selected(self, project_id: str):
        """项目选择"""
        project = self.project_manager.get_project(project_id)
        if project:
            self._load_project(project)
    
    def _on_refresh_project(self):
        """刷新当前项目"""
        if self.current_project:
            self._load_project(self.current_project)
    
    def _load_project(self, project: DatasetProject):
        """加载项目"""
        # 取消之前的任务
        if self._scanner and self._scanner.isRunning():
            self._scanner.cancel()
            self._scanner.wait()
        
        if self._thumbnail_loader and self._thumbnail_loader.isRunning():
            self._thumbnail_loader.cancel()
            self._thumbnail_loader.wait()
        
        self.current_project = project
        self.image_infos.clear()
        self.image_paths.clear()
        self.filtered_image_paths.clear()
        self.image_list_panel.clear()
        self.preview_widget.set_image(None)
        self.statistics_panel.clear()
        
        # 检查目录是否存在
        if not os.path.isdir(project.directory):
            InfoBar.error(
                title="错误",
                content=f"目录不存在: {project.directory}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            self.status_label.setText("目录不存在")
            return
        
        # 更新UI状态
        self.refresh_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"正在扫描: {project.name}...")
        
        # 启动扫描
        self._scanner = ImageScanner(project.directory)
        self._scanner.progress.connect(self._on_scan_progress)
        self._scanner.finished.connect(self._on_scan_finished)
        self._scanner.start()
    
    def _on_scan_progress(self, current: int, total: int):
        """扫描进度"""
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.status_label.setText(f"正在扫描: {current}/{total}")
    
    def _on_scan_finished(self, image_infos: List[ImageInfo], stats: AnnotationStats):
        """扫描完成"""
        self.image_infos = sorted(image_infos, key=lambda i: i.path)
        self.image_paths = [info.path for info in self.image_infos]
        count = len(self.image_paths)
        
        # 更新统计面板
        self.statistics_panel.set_stats(stats)
        
        # 更新项目信息
        if self.current_project:
            self.current_project.image_count = count
            self.current_project.annotated_count = stats.annotated_images
            self.project_manager.update_project(self.current_project)
            self.project_list_widget.update_project_item(self.current_project)
        
        if count == 0:
            self.progress_bar.setVisible(False)
            self.status_label.setText("未找到图片文件")
            self.annotate_btn.setEnabled(False)
            InfoBar.info(
                title="提示",
                content="该目录下未找到支持的图片文件",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        # 更新筛选项与图片列表
        image_types = sorted({info.image_type for info in self.image_infos if info.image_type})
        self.image_list_panel.set_type_options(image_types)
        self._apply_filters()
        self.status_label.setText(f"扫描完成，共 {count} 张图片，正在加载缩略图...")
        
        # 启动缩略图加载
        self._thumbnail_loader = ThumbnailLoader(self.image_paths)
        self._thumbnail_loader.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self._thumbnail_loader.all_loaded.connect(self._on_thumbnails_all_loaded)
        self._thumbnail_loader.start()
    
    def _on_thumbnail_loaded(self, path: str, image: QImage):
        """缩略图加载 - 在主线程中将 QImage 转换为 QPixmap"""
        pixmap = QPixmap.fromImage(image)
        self.image_list_panel.update_thumbnail(path, pixmap)
        
        # 更新进度
        loaded = len(self.image_list_panel._thumbnail_cache)
        total = len(self.image_paths)
        if total > 0:
            self.progress_bar.setValue(int(loaded / total * 100))
    
    def _on_thumbnails_all_loaded(self):
        """所有缩略图加载完成"""
        self.progress_bar.setVisible(False)
        project_name = self.current_project.name if self.current_project else ""
        self.status_label.setText(f"{project_name}: 已加载 {len(self.image_paths)} 张图片")
        
        InfoBar.success(
            title="完成",
            content=f"成功加载 {len(self.image_paths)} 张图片",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )

    def _on_filters_changed(self, annotation_filter: str, image_type_filter: str):
        """筛选变化"""
        del annotation_filter, image_type_filter
        self._apply_filters()

    def _apply_filters(self):
        """根据筛选条件刷新列表"""
        annotation_filter = self.image_list_panel.annotation_filter.currentText()
        type_filter = self.image_list_panel.type_filter.currentText().lower()

        filtered = []
        for info in self.image_infos:
            if annotation_filter == "已标注" and not info.is_annotated:
                continue
            if annotation_filter == "未标注" and info.is_annotated:
                continue
            if type_filter != "全部" and info.image_type != type_filter:
                continue
            filtered.append(info.path)

        self.filtered_image_paths = filtered
        self.image_list_panel.set_images(self.filtered_image_paths)
        self.preview_widget.set_image(None)
        self.annotate_btn.setEnabled(False)

        if self.current_project:
            self.status_label.setText(
                f"{self.current_project.name}: 共 {len(self.image_paths)} 张，筛选后 {len(self.filtered_image_paths)} 张"
            )
    
    def _on_image_selected(self, image_path: str):
        """图片选择"""
        self.preview_widget.set_image(image_path)
        self.annotate_btn.setEnabled(True)
    
    def _on_image_double_clicked(self, image_path: str):
        """图片双击"""
        self._open_for_annotation(image_path)
    
    def _on_annotate_clicked(self):
        """打开标注"""
        path = self.image_list_panel.get_selected_path()
        if path:
            self._open_for_annotation(path)
    
    def _open_for_annotation(self, image_path: str):
        """打开图片进行标注"""
        if not os.path.exists(image_path):
            InfoBar.error(
                title="错误",
                content="图片文件不存在",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        # 获取项目目录
        directory = ""
        if self.current_project:
            directory = self.current_project.directory
        else:
            # 如果没有项目，使用图片所在目录
            directory = str(Path(image_path).parent)
        
        self.request_annotation.emit(directory, image_path)
    
    def get_selected_image_path(self) -> Optional[str]:
        """获取当前选中的图片路径"""
        return self.image_list_panel.get_selected_path()
    
    def get_all_image_paths(self) -> List[str]:
        """获取所有图片路径"""
        return self.image_paths.copy()
    
    def get_current_project(self) -> Optional[DatasetProject]:
        """获取当前项目"""
        return self.current_project
