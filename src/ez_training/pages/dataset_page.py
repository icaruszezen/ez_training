"""
数据集管理页面
功能：项目管理、导入、目录扫描、数据预览、标注联动
"""

import json
import logging
import os
import tempfile
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QColor, QFont, QPainter, QPen, QPalette
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
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QApplication,
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
    SwitchButton,
)

from ez_training.common.annotation_utils import (
    parse_yolo_labels,
    parse_voc_labels,
    read_annotation_boxes,
)
from ez_training.common.constants import SUPPORTED_IMAGE_FORMATS, get_config_dir
from ez_training.ui.painting import draw_box_label
from ez_training.ui.workers import ThumbnailLoader

logger = logging.getLogger(__name__)


def _get_projects_file() -> Path:
    return get_config_dir() / "datasets.json"


_SUBFOLDER_SEP = "::sub::"
_ARCHIVE_SEP = "::arc::"
_ARCHIVE_PREFIX = "archive::"


@dataclass
class DatasetArchive:
    """数据集归档 —— 将多个数据集逻辑合并为一组"""
    id: str
    name: str
    project_ids: List[str]
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create(cls, name: str, project_ids: List[str]) -> "DatasetArchive":
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            project_ids=list(project_ids),
            created_at=now,
            updated_at=now,
        )


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
    subfolder: Optional[str] = None
    parent_id: Optional[str] = None
    archive_id: Optional[str] = None
    is_archive_root: bool = False

    @property
    def is_virtual(self) -> bool:
        return self.subfolder is not None and self.parent_id is not None

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
        self.archives: Dict[str, DatasetArchive] = {}
        self._subfolder_modes: Dict[str, bool] = {}
        self._subfolder_cache: Dict[str, List[str]] = {}
        self._subfolder_counts: Dict[str, Dict[str, int]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """加载项目配置"""
        config_file = _get_projects_file()
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    known_fields = {f.name for f in DatasetProject.__dataclass_fields__.values()}
                    for item in data.get("projects", []):
                        if item.get("id", "").startswith(_ARCHIVE_PREFIX):
                            continue
                        filtered = {k: v for k, v in item.items() if k in known_fields}
                        proj = DatasetProject(**filtered)
                        self.projects[proj.id] = proj
                    self._subfolder_modes = data.get("subfolder_modes", {})
                    archive_fields = {f.name for f in DatasetArchive.__dataclass_fields__.values()}
                    for item in data.get("archives", []):
                        filtered = {k: v for k, v in item.items() if k in archive_fields}
                        arc = DatasetArchive(**filtered)
                        self.archives[arc.id] = arc
            except Exception:
                logger.exception("Failed to load project config from %s", config_file)

    def _save(self):
        """保存项目配置（排除虚拟条目），使用原子写入防止中途崩溃导致 JSON 损坏"""
        config_file = _get_projects_file()
        real_projects = [p for p in self.projects.values()
                         if not p.is_virtual and not p.id.startswith(_ARCHIVE_PREFIX)]
        serialised = []
        for p in real_projects:
            d = asdict(p)
            d.pop("subfolder", None)
            d.pop("parent_id", None)
            d.pop("archive_id", None)
            d.pop("is_archive_root", None)
            serialised.append(d)
        archive_list = [asdict(a) for a in self.archives.values()]
        data = {
            "projects": serialised,
            "subfolder_modes": self._subfolder_modes,
            "archives": archive_list,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(config_file.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(config_file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_project(self, project: DatasetProject):
        """添加项目"""
        self.projects[project.id] = project
        self._save()

    def remove_project(self, project_id: str):
        """删除项目，同时从所有归档中移除该成员"""
        if project_id in self.projects:
            del self.projects[project_id]
            self._subfolder_modes.pop(project_id, None)
            self._subfolder_cache.pop(project_id, None)
            empty_archives: List[str] = []
            for arc in self.archives.values():
                if project_id in arc.project_ids:
                    arc.project_ids.remove(project_id)
                    if not arc.project_ids:
                        empty_archives.append(arc.id)
            for arc_id in empty_archives:
                del self.archives[arc_id]
            self._save()

    def update_project(self, project: DatasetProject):
        """更新项目"""
        if project.is_virtual or project.archive_id is not None:
            return
        project.updated_at = datetime.now().isoformat()
        self.projects[project.id] = project
        self._save()

    # ------------------------------------------------------------------
    # Archive CRUD
    # ------------------------------------------------------------------

    def add_archive(self, archive: DatasetArchive):
        self.archives[archive.id] = archive
        self._save()

    def remove_archive(self, archive_id: str):
        if archive_id in self.archives:
            del self.archives[archive_id]
            self._save()

    def get_archive(self, archive_id: str) -> Optional[DatasetArchive]:
        return self.archives.get(archive_id)

    def get_all_archives(self) -> List[DatasetArchive]:
        return list(self.archives.values())

    def clean_orphan_archive_entries(self):
        """删除 self.projects 中 ID 以 archive:: 开头的孤儿条目并保存"""
        orphans = [pid for pid in self.projects if pid.startswith(_ARCHIVE_PREFIX)]
        if orphans:
            for pid in orphans:
                del self.projects[pid]
            self._save()

    def get_real_projects(self, exclude_archived: bool = False) -> List[DatasetProject]:
        """仅返回真实（非虚拟）项目列表，用于归档对话框等。

        当 *exclude_archived* 为 True 时，已经属于某个归档的项目不会返回。
        """
        archived_ids: set = set()
        if exclude_archived:
            for arc in self.archives.values():
                archived_ids.update(arc.project_ids)
        return [p for p in self.projects.values()
                if not p.is_virtual and p.id not in archived_ids]

    def get_archive_directories(self, archive_id: str) -> List[str]:
        """返回归档内所有成员的目录列表"""
        arc = self.archives.get(archive_id)
        if arc is None:
            return []
        dirs: List[str] = []
        for pid in arc.project_ids:
            proj = self.projects.get(pid)
            if proj and os.path.isdir(proj.directory):
                dirs.append(proj.directory)
        return dirs

    def get_directories(self, project_id: str) -> List[str]:
        """获取项目对应的目录列表（归档聚合 -> 多目录, 其他 -> 单目录）"""
        proj = self.get_project(project_id)
        if proj is None:
            return []
        if proj.is_archive_root and proj.archive_id:
            return self.get_archive_directories(proj.archive_id)
        if proj.directory and os.path.isdir(proj.directory):
            return [proj.directory]
        return []

    # ------------------------------------------------------------------
    # Query (with virtual subfolder / archive expansion)
    # ------------------------------------------------------------------

    def get_project(self, project_id: str) -> Optional[DatasetProject]:
        """获取项目，支持虚拟子文件夹 ID 和归档虚拟 ID"""
        if project_id.startswith(_ARCHIVE_PREFIX):
            return self._resolve_archive_entry(project_id)
        if _SUBFOLDER_SEP in project_id:
            parent_id, subfolder = project_id.split(_SUBFOLDER_SEP, 1)
            parent = self.projects.get(parent_id)
            if parent is None:
                return None
            if parent_id not in self._subfolder_counts:
                self.detect_subfolders(parent_id)
            return self._make_virtual(parent, subfolder)
        return self.projects.get(project_id)

    def get_all_projects(self, exclude_archived: bool = False) -> List[DatasetProject]:
        """获取所有项目；启用子文件夹模式的项目会展开为虚拟条目；归档追加在末尾。

        当 *exclude_archived* 为 True 时，已被归档的数据集不再作为独立条目
        出现——它们只在归档分组内可见，避免 ComboBox 出现重复选项。
        """
        archived_ids: set = set()
        if exclude_archived:
            for arc in self.archives.values():
                archived_ids.update(arc.project_ids)

        result: List[DatasetProject] = []
        for proj in self.projects.values():
            if proj.id in archived_ids:
                continue
            if self._subfolder_modes.get(proj.id, False):
                subfolders = self.detect_subfolders(proj.id)
                for sf in subfolders:
                    result.append(self._make_virtual(proj, sf))
            else:
                result.append(proj)
        for arc in self.archives.values():
            result.append(self._make_archive_root(arc))
            for pid in arc.project_ids:
                member = self.projects.get(pid)
                if member:
                    result.append(self._make_archive_member(arc, member))
        return result

    # ------------------------------------------------------------------
    # Subfolder mode helpers
    # ------------------------------------------------------------------

    def detect_subfolders(self, project_id: str) -> List[str]:
        """检测项目目录的直接子文件夹（含图片的），同时统计图片数，缓存结果"""
        cached = self._subfolder_cache.get(project_id)
        if cached is not None:
            return cached

        proj = self.projects.get(project_id)
        if proj is None or not os.path.isdir(proj.directory):
            return []

        base = Path(proj.directory)
        subfolders: List[str] = []
        counts: Dict[str, int] = {}

        try:
            root_count = sum(
                1 for f in base.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
            )
        except OSError:
            root_count = 0
        if root_count > 0:
            subfolders.append("(root)")
            counts["(root)"] = root_count

        try:
            entries = sorted(base.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            try:
                n = sum(
                    1 for _, _, files in os.walk(str(entry))
                    for fn in files
                    if Path(fn).suffix.lower() in SUPPORTED_IMAGE_FORMATS
                )
            except OSError:
                n = 0
            if n > 0:
                subfolders.append(entry.name)
                counts[entry.name] = n

        self._subfolder_cache[project_id] = subfolders
        self._subfolder_counts[project_id] = counts
        return subfolders

    def has_subfolders(self, project_id: str) -> bool:
        subs = self.detect_subfolders(project_id)
        if not subs:
            return False
        if subs == ["(root)"]:
            return False
        return True

    def set_subfolder_mode(self, project_id: str, enabled: bool):
        self._subfolder_modes[project_id] = enabled
        self._save()

    def is_subfolder_mode(self, project_id: str) -> bool:
        return self._subfolder_modes.get(project_id, False)

    def clear_subfolder_cache(self, project_id: Optional[str] = None):
        if project_id:
            self._subfolder_cache.pop(project_id, None)
            self._subfolder_counts.pop(project_id, None)
        else:
            self._subfolder_cache.clear()
            self._subfolder_counts.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_virtual(self, parent: DatasetProject, subfolder: str) -> DatasetProject:
        """根据父项目和子文件夹名生成虚拟条目"""
        if subfolder == "(root)":
            directory = parent.directory
        else:
            directory = str(Path(parent.directory) / subfolder)
        counts = self._subfolder_counts.get(parent.id, {})
        image_count = counts.get(subfolder, 0)
        return DatasetProject(
            id=f"{parent.id}{_SUBFOLDER_SEP}{subfolder}",
            name=f"{parent.name}/{subfolder}",
            directory=directory,
            image_count=image_count,
            annotated_count=0,
            created_at=parent.created_at,
            updated_at=parent.updated_at,
            subfolder=subfolder,
            parent_id=parent.id,
        )

    def _make_archive_root(self, archive: DatasetArchive) -> DatasetProject:
        """生成归档聚合虚拟条目（代表整个归档）"""
        total_images = 0
        total_annotated = 0
        for pid in archive.project_ids:
            proj = self.projects.get(pid)
            if proj:
                total_images += proj.image_count
                total_annotated += proj.annotated_count
        return DatasetProject(
            id=f"{_ARCHIVE_PREFIX}{archive.id}",
            name=f"[归档] {archive.name}",
            directory="",
            image_count=total_images,
            annotated_count=total_annotated,
            created_at=archive.created_at,
            updated_at=archive.updated_at,
            is_archive_root=True,
            archive_id=archive.id,
        )

    def _make_archive_member(self, archive: DatasetArchive, member: DatasetProject) -> DatasetProject:
        """生成归档成员虚拟条目"""
        return DatasetProject(
            id=f"{_ARCHIVE_PREFIX}{archive.id}{_ARCHIVE_SEP}{member.id}",
            name=f"  └ {archive.name}/{member.name}",
            directory=member.directory,
            image_count=member.image_count,
            annotated_count=member.annotated_count,
            created_at=member.created_at,
            updated_at=member.updated_at,
            archive_id=archive.id,
        )

    def extract_archive_id(self, project_id: str) -> Optional[str]:
        """从归档虚拟 ID 中提取 archive_id，非归档 ID 返回 None"""
        if not project_id.startswith(_ARCHIVE_PREFIX):
            return None
        body = project_id[len(_ARCHIVE_PREFIX):]
        if _ARCHIVE_SEP in body:
            return body.split(_ARCHIVE_SEP, 1)[0]
        return body

    def _resolve_archive_entry(self, project_id: str) -> Optional[DatasetProject]:
        """解析归档虚拟 ID -> DatasetProject"""
        body = project_id[len(_ARCHIVE_PREFIX):]
        if _ARCHIVE_SEP in body:
            arc_id, member_id = body.split(_ARCHIVE_SEP, 1)
            arc = self.archives.get(arc_id)
            member = self.projects.get(member_id)
            if arc and member:
                return self._make_archive_member(arc, member)
            return None
        arc = self.archives.get(body)
        if arc:
            return self._make_archive_root(arc)
        return None


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
    """异步图片扫描线程，支持单目录或多目录扫描"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list, object)  # image_infos, AnnotationStats
    
    def __init__(self, directory: str = "", classes_file: str = None,
                 recursive: bool = True, directories: List[str] = None):
        super().__init__()
        if directories:
            self._directories = list(directories)
        else:
            self._directories = [directory] if directory else []
        self.directory = self._directories[0] if self._directories else ""
        self.classes_file = classes_file
        self.recursive = recursive
        self._cancelled = False
        self._class_names: List[str] = []
    
    def _load_classes(self):
        """加载 YOLO classes.txt 文件"""
        possible_paths: List[Path] = []
        if self.classes_file:
            possible_paths.append(Path(self.classes_file))
        for d in self._directories:
            possible_paths.extend([
                Path(d) / "classes.txt",
                Path(d) / "labels" / "classes.txt",
                Path(d) / ".." / "classes.txt",
            ])
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self._class_names = [line.strip() for line in f if line.strip()]
                    return
                except Exception:
                    logger.debug("Failed to load classes from %s", path)
    
    def run(self):
        image_infos = []
        all_files = []
        stats = AnnotationStats()
        label_counter = Counter()
        
        self._load_classes()
        
        for directory in self._directories:
            if self._cancelled:
                break
            if not os.path.isdir(directory):
                continue
            try:
                if self.recursive:
                    for root, _, files in os.walk(directory):
                        if self._cancelled:
                            break
                        for file in files:
                            if Path(file).suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                                all_files.append(os.path.join(root, file))
                else:
                    for file in os.listdir(directory):
                        full = os.path.join(directory, file)
                        if os.path.isfile(full) and Path(file).suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                            all_files.append(full)
            except OSError as e:
                logger.warning("Error scanning directory %s: %s", directory, e)
        
        total = len(all_files)
        stats.total_images = total
        
        for i, file_path in enumerate(all_files):
            if self._cancelled:
                break
            
            path = Path(file_path)
            labels = []
            
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                labels = parse_yolo_labels(txt_path, self._class_names)
                if not labels:
                    xml_path = path.with_suffix(".xml")
                    if xml_path.exists():
                        labels = parse_voc_labels(xml_path)
            else:
                xml_path = path.with_suffix(".xml")
                if xml_path.exists():
                    labels = parse_voc_labels(xml_path)
            
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
        self._cancelled = True


class ProjectListWidget(CardWidget):
    """项目列表组件"""
    
    project_selected = pyqtSignal(str)  # project_id
    project_deleted = pyqtSignal(str)   # project_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
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
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 8px;
                outline: none;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid palette(midlight);
            }
            QListWidget::item:last-child {
                border-bottom: none;
            }
            QListWidget::item:selected {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
            QListWidget::item:hover {
                background-color: palette(alternate-base);
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
    
    @staticmethod
    def _format_project_text(project: DatasetProject) -> str:
        if project.is_archive_root:
            return (
                f"{project.name}\n"
                f"📦 {project.image_count} 张图片 · ✅ {project.annotated_count} 已标注"
            )
        if project.archive_id and not project.is_archive_root:
            return (
                f"{project.name}\n"
                f"📁 {project.directory}\n"
                f"🖼 {project.image_count} 张"
            )
        text = f"{project.name}\n"
        text += f"📁 {project.directory}\n"
        text += f"🖼 {project.image_count} 张图片 · ✅ {project.annotated_count} 已标注"
        return text

    def add_project_item(self, project: DatasetProject, select: bool = False):
        """添加项目项"""
        item = QListWidgetItem()
        item.setData(Qt.UserRole, project.id)
        item.setText(self._format_project_text(project))

        if project.is_archive_root:
            hl = QApplication.palette().color(QPalette.Highlight)
            hl.setAlpha(30)
            item.setBackground(hl)
            item.setSizeHint(QSize(0, 56))
        elif project.archive_id:
            item.setBackground(QApplication.palette().color(QPalette.AlternateBase))
            item.setSizeHint(QSize(0, 68))
        else:
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
                item.setText(self._format_project_text(project))
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

    def select_project(self, project_id: str):
        """按 ID 选中项目（不触发 project_selected 信号）"""
        self.project_list.blockSignals(True)
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.data(Qt.UserRole) == project_id:
                self.project_list.setCurrentItem(item)
                is_subfolder = _SUBFOLDER_SEP in project_id
                is_archive_member = (
                    project_id.startswith(_ARCHIVE_PREFIX) and _ARCHIVE_SEP in project_id
                )
                self.delete_btn.setEnabled(not is_subfolder and not is_archive_member)
                break
        self.project_list.blockSignals(False)

    def _on_item_clicked(self, item: QListWidgetItem):
        """项目点击"""
        project_id = item.data(Qt.UserRole)
        is_subfolder = _SUBFOLDER_SEP in project_id
        is_archive_member = (
            project_id.startswith(_ARCHIVE_PREFIX) and _ARCHIVE_SEP in project_id
        )
        is_archive_root = (
            project_id.startswith(_ARCHIVE_PREFIX) and _ARCHIVE_SEP not in project_id
        )
        can_delete = not is_subfolder and not is_archive_member
        self.delete_btn.setEnabled(can_delete)
        if is_archive_root:
            self.delete_btn.setText("解散归档")
        else:
            self.delete_btn.setText("删除")
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
                background-color: palette(alternate-base);
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
                border: 1px solid palette(mid);
                border-radius: 6px;
                background-color: palette(base);
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
            no_label.setStyleSheet("color: palette(mid);")
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
            name_label.setStyleSheet("color: palette(text);")
            row_layout.addWidget(name_label, 1)
            
            # 数量和百分比
            count_label = CaptionLabel(f"{count} ({percentage:.1f}%)")
            count_label.setStyleSheet("color: palette(dark);")
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


class _AnnotatedPreviewWorker(QThread):
    """Background worker that loads an image, draws annotation overlays, and scales."""

    finished = pyqtSignal(str, QImage)

    _LABEL_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0),
        (128, 0, 128), (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 0),
    ]

    def __init__(self, image_path: str, class_names: List[str],
                 target_w: int, target_h: int):
        super().__init__()
        self._image_path = image_path
        self._class_names = list(class_names)
        self._target_w = target_w
        self._target_h = target_h
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self._cancelled:
            return
        reader = QImageReader(self._image_path)
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if not orig_size.isValid() or self._cancelled:
            return

        img_w, img_h = orig_size.width(), orig_size.height()
        annotations = read_annotation_boxes(
            self._image_path, img_w, img_h, self._class_names,
        )
        if self._cancelled:
            return

        if annotations:
            image = reader.read()
            if image.isNull() or self._cancelled:
                return
            self._draw_on_image(image, annotations)
            if self._cancelled:
                return
            result = image.scaled(self._target_w, self._target_h,
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            target = QSize(self._target_w, self._target_h)
            load_size = orig_size.scaled(target, Qt.KeepAspectRatio)
            reader.setScaledSize(load_size)
            result = reader.read()
            if result.isNull():
                return

        if not self._cancelled:
            self.finished.emit(self._image_path, result)

    def _draw_on_image(self, image: QImage, annotations: List[dict]):
        label_set = sorted({a["label"] for a in annotations})
        color_map = {
            lbl: self._LABEL_COLORS[i % len(self._LABEL_COLORS)]
            for i, lbl in enumerate(label_set)
        }
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        for ann in annotations:
            r, g, b = color_map[ann["label"]]
            pen = QPen(QColor(r, g, b))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(ann["xmin"], ann["ymin"],
                             ann["xmax"] - ann["xmin"],
                             ann["ymax"] - ann["ymin"])
        painter.end()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Microsoft YaHei", -1)
        font.setPixelSize(16)
        painter.setFont(font)
        for ann in annotations:
            bgr = tuple(reversed(color_map[ann["label"]]))
            draw_box_label(painter, ann["label"],
                           ann["xmin"], ann["ymin"], bgr)
        painter.end()


class ImagePreviewWidget(QFrame):
    """图片预览组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(280)
        self._setup_ui()
        self._current_path: Optional[str] = None
        self._project_directory: Optional[str] = None
        self._class_names: List[str] = []
        self._preview_worker: Optional[_AnnotatedPreviewWorker] = None
        self._preview_generation = 0
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 标题行：标题 + 显示标注开关
        header_layout = QHBoxLayout()
        self.title_label = SubtitleLabel("图片预览")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.annotation_switch = SwitchButton("标注")
        self.annotation_switch.setChecked(False)
        self.annotation_switch.checkedChanged.connect(self._on_annotation_toggle)
        header_layout.addWidget(self.annotation_switch)
        layout.addLayout(header_layout)
        
        # 图片预览区域
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(200, 200)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: palette(alternate-base);
                border: 1px solid palette(mid);
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
        self._cancel_preview_worker()
        
        if not image_path or not os.path.exists(image_path):
            self.preview_label.setText("选择图片以预览")
            self._clear_info()
            return
        
        reader = QImageReader(image_path)
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if not orig_size.isValid():
            self.preview_label.setText("无法加载图片")
            self._clear_info()
            return
        
        img_w, img_h = orig_size.width(), orig_size.height()
        preview_size = self.preview_label.size()
        target_w = max(preview_size.width() - 20, 1)
        target_h = max(preview_size.height() - 20, 1)

        if self.annotation_switch.isChecked():
            self._preview_generation += 1
            gen = self._preview_generation
            self._preview_worker = _AnnotatedPreviewWorker(
                image_path, self._class_names, target_w, target_h,
            )
            self._preview_worker.finished.connect(
                lambda p, img, _g=gen: self._on_preview_ready(p, img, _g)
            )
            self._preview_worker.start()
        else:
            target = QSize(target_w, target_h)
            load_size = orig_size.scaled(target, Qt.KeepAspectRatio)
            reader.setScaledSize(load_size)
            image = reader.read()
            if image.isNull():
                self.preview_label.setText("无法加载图片")
                self._clear_info()
                return
            self.preview_label.setPixmap(QPixmap.fromImage(image))
        
        file_path = Path(image_path)
        try:
            file_stat = file_path.stat()
            self.file_size_label.setText(f"大小: {self._format_size(file_stat.st_size)}")
        except OSError:
            self.file_size_label.setText("大小: -")
        
        self.file_name_label.setText(f"文件名: {file_path.name}")
        self.image_size_label.setText(f"尺寸: {img_w} × {img_h}")
        
        annotation_status = self._check_annotation_status(image_path)
        self.annotation_status_label.setText(f"标注: {annotation_status}")

    def _cancel_preview_worker(self):
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.cancel()

    def _on_preview_ready(self, path: str, image: QImage, generation: int):
        if generation != self._preview_generation or path != self._current_path:
            return
        self.preview_label.setPixmap(QPixmap.fromImage(image))
    
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
        try:
            path = Path(image_path)
            
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        lines = [l for l in f.readlines() if l.strip()]
                        if lines:
                            return f"已标注 (YOLO, {len(lines)} 个对象)"
                except (OSError, UnicodeDecodeError):
                    pass
            
            xml_path = path.with_suffix(".xml")
            if xml_path.exists():
                return "已标注 (VOC)"
        except Exception:
            logger.debug("Failed to check annotation status for %s", image_path)
        
        return "未标注"
    
    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    @property
    def project_directory(self) -> Optional[str]:
        return self._project_directory

    @project_directory.setter
    def project_directory(self, value: Optional[str]):
        if value == self._project_directory:
            return
        self._project_directory = value
        self._class_names = []
        if value:
            self._load_class_names(value)

    def _load_class_names(self, directory: str):
        """加载 YOLO classes.txt（搜索路径同 ImageScanner）"""
        candidates = [
            Path(directory) / "classes.txt",
            Path(directory) / "labels" / "classes.txt",
            Path(directory) / ".." / "classes.txt",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self._class_names = [line.strip() for line in f if line.strip()]
                    return
                except Exception:
                    logger.debug("Failed to load class names from %s", p)

    def _on_annotation_toggle(self, checked: bool):
        """开关切换时刷新预览"""
        del checked
        if self._current_path:
            self.set_image(self._current_path)


class ImageListPanel(CardWidget):
    """图片列表面板"""
    
    image_selected = pyqtSignal(str)  # image_path
    image_double_clicked = pyqtSignal(str)  # image_path
    filters_changed = pyqtSignal(str, str)  # annotation_filter, type_filter
    page_changed = pyqtSignal(list)  # 当前页的路径列表
    
    PAGE_SIZE = 200
    THUMBNAIL_CACHE_MAX = 2000
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumbnail_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._path_to_item = {}
        self._placeholder_pixmap = None
        self._all_paths: List[str] = []
        self._current_page = 0
        self._total_pages = 0
        self._setup_ui()
    
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
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setMovement(QListWidget.Static)
        self.image_list.setUniformItemSizes(True)  # 优化：统一项目大小
        self.image_list.setStyleSheet("""
            QListWidget {
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 8px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 6px;
                color: palette(text);
            }
            QListWidget::item:selected {
                background-color: palette(highlight);
                border: 2px solid palette(highlight);
                color: palette(highlighted-text);
            }
            QListWidget::item:hover {
                background-color: palette(alternate-base);
            }
        """)
        self.image_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.image_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.image_list, 1)
        
        # 分页控件
        page_layout = QHBoxLayout()
        page_layout.setSpacing(8)
        page_layout.addStretch()
        self.prev_page_btn = PushButton("上一页", self, FIF.LEFT_ARROW)
        self.prev_page_btn.setFixedHeight(30)
        self.prev_page_btn.clicked.connect(self._on_prev_page)
        page_layout.addWidget(self.prev_page_btn)
        self.page_label = CaptionLabel("第 0/0 页")
        page_layout.addWidget(self.page_label)
        self.next_page_btn = PushButton("下一页", self, FIF.RIGHT_ARROW)
        self.next_page_btn.setFixedHeight(30)
        self.next_page_btn.clicked.connect(self._on_next_page)
        page_layout.addWidget(self.next_page_btn)
        page_layout.addStretch()
        layout.addLayout(page_layout)
    
    def set_images(self, image_paths: List[str], reset_cache: bool = False):
        """设置图片列表（分页模式，仅显示当前页）"""
        self.image_list.clear()
        self._path_to_item.clear()
        if reset_cache:
            self._thumbnail_cache.clear()
        
        self._all_paths = list(image_paths)
        total = len(self._all_paths)
        self._total_pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE) if total else 0
        self._current_page = 0
        
        if self._placeholder_pixmap is None:
            self._placeholder_pixmap = self._create_placeholder_pixmap()
        
        if total == 0:
            self.count_label.setText("共 0 张图片")
            self._update_page_controls()
            return
        
        self._show_page(0)
    
    def _show_page(self, page: int):
        """显示指定页的图片"""
        self._current_page = page
        self.image_list.clear()
        self._path_to_item.clear()
        
        start = page * self.PAGE_SIZE
        end = min(start + self.PAGE_SIZE, len(self._all_paths))
        page_paths = self._all_paths[start:end]
        
        for path in page_paths:
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
        
        total = len(self._all_paths)
        self.count_label.setText(
            f"共 {total} 张图片 (第 {page + 1}/{self._total_pages} 页)"
        )
        self._update_page_controls()
        self.page_changed.emit(page_paths)
    
    def _update_page_controls(self):
        """更新分页按钮和标签状态"""
        has_pages = self._total_pages > 0
        self.prev_page_btn.setEnabled(has_pages and self._current_page > 0)
        self.next_page_btn.setEnabled(has_pages and self._current_page < self._total_pages - 1)
        if has_pages:
            self.page_label.setText(f"第 {self._current_page + 1}/{self._total_pages} 页")
        else:
            self.page_label.setText("第 0/0 页")
    
    def _on_prev_page(self):
        if self._current_page > 0:
            self._show_page(self._current_page - 1)
    
    def _on_next_page(self):
        if self._current_page < self._total_pages - 1:
            self._show_page(self._current_page + 1)
    
    def update_thumbnail(self, path: str, pixmap: QPixmap):
        """更新缩略图 - 使用 LRU OrderedDict，超限时淘汰最旧条目"""
        if path in self._thumbnail_cache:
            self._thumbnail_cache.move_to_end(path)
        self._thumbnail_cache[path] = pixmap
        while len(self._thumbnail_cache) > self.THUMBNAIL_CACHE_MAX:
            self._thumbnail_cache.popitem(last=False)
        
        item = self._path_to_item.get(path)
        if item:
            item.setIcon(QIcon(pixmap))
    
    def clear(self):
        """清空列表"""
        self.image_list.clear()
        self._thumbnail_cache.clear()
        self._path_to_item.clear()
        self._all_paths.clear()
        self._current_page = 0
        self._total_pages = 0
        self.count_label.setText("共 0 张图片")
        self._update_page_controls()
        self.reset_filters()
        self.set_type_options([])

    def get_uncached_paths(self, paths: List[str]) -> List[str]:
        """Return the subset of *paths* not yet present in the thumbnail cache."""
        return [p for p in paths if p not in self._thumbnail_cache]

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

    def get_selected_paths(self) -> List[str]:
        """获取所有选中的图片路径"""
        return [item.data(Qt.UserRole) for item in self.image_list.selectedItems()]


class CreateArchiveDialog(QDialog):
    """创建数据集归档对话框"""

    def __init__(self, projects: List[DatasetProject], parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建数据集归档")
        self.setMinimumSize(420, 400)
        self._projects = projects
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(StrongBodyLabel("归档名称"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入归档名称")
        self.name_edit.textChanged.connect(self._validate)
        layout.addWidget(self.name_edit)

        layout.addWidget(StrongBodyLabel("选择数据集（至少 2 个）"))

        self.project_list = QListWidget()
        self.project_list.setSelectionMode(QAbstractItemView.NoSelection)
        for proj in self._projects:
            item = QListWidgetItem(f"{proj.name}  ({proj.image_count} 张)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, proj.id)
            self.project_list.addItem(item)
        self.project_list.itemChanged.connect(self._validate)
        layout.addWidget(self.project_list, 1)

        self.hint_label = CaptionLabel("")
        self.hint_label.setStyleSheet("color: #e53935;")
        layout.addWidget(self.hint_label)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.button(QDialogButtonBox.Ok).setText("创建")
        self.button_box.button(QDialogButtonBox.Cancel).setText("取消")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._validate()

    def _validate(self, *_args):
        name_ok = bool(self.name_edit.text().strip())
        checked = self._checked_ids()
        count_ok = len(checked) >= 2
        ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        ok_btn.setEnabled(name_ok and count_ok)
        if not name_ok:
            self.hint_label.setText("请输入归档名称")
        elif not count_ok:
            self.hint_label.setText(f"已选 {len(checked)} 个，至少选择 2 个数据集")
        else:
            self.hint_label.setText(f"已选 {len(checked)} 个数据集")
            self.hint_label.setStyleSheet("color: #43a047;")
            return
        self.hint_label.setStyleSheet("color: #e53935;")

    def _checked_ids(self) -> List[str]:
        ids: List[str] = []
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(item.data(Qt.UserRole))
        return ids

    def get_result(self):
        """返回 (名称, 选中的项目ID列表)"""
        return self.name_edit.text().strip(), self._checked_ids()


class DatasetPage(QWidget):
    """数据集管理页面"""
    
    # 信号：请求打开图片进行标注 (目录路径, 图片路径)
    request_annotation = pyqtSignal(str, str)  # directory, image_path
    # 信号：请求批量标注 (目录路径, 图片路径列表)
    request_batch_annotation = pyqtSignal(str, list)  # directory, image_paths
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_manager = ProjectManager()
        self.current_project: Optional[DatasetProject] = None
        self.image_infos: List[ImageInfo] = []
        self.image_paths: List[str] = []
        self.filtered_image_paths: List[str] = []
        self._scanner: Optional[ImageScanner] = None
        self._thumbnail_loader: Optional[ThumbnailLoader] = None
        self._scan_generation = 0
        self._thumb_generation = 0
        self._page_thumb_total = 0
        self._page_thumb_loaded = 0
        
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
        
        # 主内容区域：水平分割（左右）
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：垂直分割（数据集项目 + 统计面板，上下排列）
        left_splitter = QSplitter(Qt.Vertical)
        
        self.project_list_widget = ProjectListWidget()
        self.project_list_widget.project_selected.connect(self._on_project_selected)
        self.project_list_widget.add_btn.clicked.connect(self._on_add_project)
        self.project_list_widget.delete_btn.clicked.connect(self._on_delete_project)
        left_splitter.addWidget(self.project_list_widget)
        
        self.statistics_panel = StatisticsPanel()
        left_splitter.addWidget(self.statistics_panel)
        
        left_splitter.setSizes([480, 160])
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 1)
        
        content_splitter.addWidget(left_splitter)
        
        # 右侧：图片列表和预览（垂直分割）
        right_splitter = QSplitter(Qt.Vertical)
        self.image_list_panel = ImageListPanel()
        self.image_list_panel.image_selected.connect(self._on_image_selected)
        self.image_list_panel.image_double_clicked.connect(self._on_image_double_clicked)
        self.image_list_panel.filters_changed.connect(self._on_filters_changed)
        self.image_list_panel.page_changed.connect(self._load_page_thumbnails)
        right_splitter.addWidget(self.image_list_panel)
        self.preview_widget = ImagePreviewWidget()
        right_splitter.addWidget(self.preview_widget)
        right_splitter.setSizes([180, 420])
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        
        content_splitter.addWidget(right_splitter)
        
        content_splitter.setSizes([200, 800])
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 4)
        
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

        title = TitleLabel("数据集管理")
        layout.addWidget(title)

        layout.addStretch()

        # 子文件夹模式开关
        self.subfolder_switch = SwitchButton("子文件夹模式")
        self.subfolder_switch.setChecked(False)
        self.subfolder_switch.setEnabled(False)
        self.subfolder_switch.setVisible(False)
        self.subfolder_switch.checkedChanged.connect(self._on_subfolder_mode_changed)
        layout.addWidget(self.subfolder_switch)

        self.refresh_btn = PushButton("刷新", self, FIF.SYNC)
        self.refresh_btn.clicked.connect(self._on_refresh_project)
        self.refresh_btn.setEnabled(False)
        layout.addWidget(self.refresh_btn)

        self.annotate_btn = PrimaryPushButton("打开标注", self, FIF.EDIT)
        self.annotate_btn.setEnabled(False)
        self.annotate_btn.clicked.connect(self._on_annotate_clicked)
        layout.addWidget(self.annotate_btn)

        self.batch_annotate_btn = PushButton("批量标注", self, FIF.COPY)
        self.batch_annotate_btn.setEnabled(False)
        self.batch_annotate_btn.clicked.connect(self._on_batch_annotate_clicked)
        layout.addWidget(self.batch_annotate_btn)

        self.archive_btn = PushButton("创建归档", self, FIF.ZIP_FOLDER)
        self.archive_btn.clicked.connect(self._on_create_archive)
        layout.addWidget(self.archive_btn)

        return header
    
    def _load_projects(self):
        """加载所有项目（已归档的数据集只在归档组内显示）"""
        self.project_list_widget.clear_projects()
        projects = self.project_manager.get_all_projects(exclude_archived=True)
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
        """删除项目或解散归档"""
        project_id = self.project_list_widget.get_selected_project_id()
        if not project_id or _SUBFOLDER_SEP in project_id:
            return

        # 归档聚合条目 -> 解散归档
        if project_id.startswith(_ARCHIVE_PREFIX):
            self._on_delete_archive(project_id)
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
            self._load_projects()
            self._clear_current_if(
                lambda cur: cur.id == project_id or cur.parent_id == project_id
            )
            InfoBar.success(
                title="成功",
                content=f"已删除项目: {project.name}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

    def _on_create_archive(self):
        """打开创建归档对话框"""
        real_projects = self.project_manager.get_real_projects(exclude_archived=True)
        if len(real_projects) < 2:
            InfoBar.warning(
                title="提示",
                content="至少需要 2 个数据集才能创建归档",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self,
            )
            return

        dialog = CreateArchiveDialog(real_projects, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        name, project_ids = dialog.get_result()
        if not name or len(project_ids) < 2:
            return

        existing_names = {a.name for a in self.project_manager.archives.values()}
        if name in existing_names:
            InfoBar.warning(
                title="名称重复",
                content=f"归档名称 \"{name}\" 已存在，请使用其他名称",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self,
            )
            return

        archive = DatasetArchive.create(name, project_ids)
        self.project_manager.add_archive(archive)
        self._load_projects()

        InfoBar.success(
            title="成功",
            content=f"已创建归档: {name} ({len(project_ids)} 个数据集)",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self,
        )

    def _on_delete_archive(self, project_id: str):
        """解散归档"""
        project = self.project_manager.get_project(project_id)
        arc_id_from_pid = self.project_manager.extract_archive_id(project_id)

        arc = None
        arc_name = ""
        if project and project.archive_id:
            arc = self.project_manager.get_archive(project.archive_id)
        if not arc and arc_id_from_pid:
            arc = self.project_manager.get_archive(arc_id_from_pid)

        if arc:
            arc_name = arc.name
        else:
            orphan = self.project_manager.projects.get(project_id)
            arc_name = orphan.name if orphan else project_id

        reply = QMessageBox.question(
            self,
            "解散归档",
            f"确定要解散归档 \"{arc_name}\" 吗？\n\n（成员数据集不受影响）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            if arc:
                self.project_manager.remove_archive(arc.id)
            self.project_manager.clean_orphan_archive_entries()
            self._load_projects()
            effective_arc_id = arc.id if arc else arc_id_from_pid

            def _belongs_to_disbanded_archive(cur):
                if cur.archive_id == effective_arc_id:
                    return True
                parsed = self.project_manager.extract_archive_id(cur.id)
                return parsed is not None and not self.project_manager.get_archive(parsed)

            self._clear_current_if(_belongs_to_disbanded_archive)
            InfoBar.success(
                title="成功",
                content=f"已解散归档: {arc_name}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self,
            )

    def _clear_current_if(self, predicate):
        """若当前项目满足 predicate 则清空中间面板"""
        cur = self.current_project
        if cur is not None and predicate(cur):
            self.current_project = None
            self.image_infos.clear()
            self.image_paths.clear()
            self.filtered_image_paths.clear()
            self.image_list_panel.clear()
            self.preview_widget.set_image(None)
            self.statistics_panel.clear()
            self.refresh_btn.setEnabled(False)
            self.annotate_btn.setEnabled(False)
            self.batch_annotate_btn.setEnabled(False)
            self.subfolder_switch.setVisible(False)
            self.subfolder_switch.setEnabled(False)
            self.status_label.setText("选择或创建数据集项目")
    
    def _on_project_selected(self, project_id: str):
        """项目选择"""
        project = self.project_manager.get_project(project_id)
        if project:
            self._load_project(project)
    
    def _on_refresh_project(self):
        """刷新当前项目"""
        if self.current_project:
            real_id = (
                self.current_project.parent_id
                if self.current_project.is_virtual
                else self.current_project.id
            )
            self.project_manager.clear_subfolder_cache(real_id)
            self._load_project(self.current_project)

    def _on_subfolder_mode_changed(self, checked: bool):
        """子文件夹模式开关切换"""
        if not self.current_project:
            return

        real_id = (
            self.current_project.parent_id
            if self.current_project.is_virtual
            else self.current_project.id
        )
        self.project_manager.set_subfolder_mode(real_id, checked)

        # 刷新左侧项目列表
        self._load_projects()

        if checked:
            # 自动选中第一个子文件夹条目
            subfolders = self.project_manager.detect_subfolders(real_id)
            if subfolders:
                first_id = f"{real_id}{_SUBFOLDER_SEP}{subfolders[0]}"
                self.project_list_widget.select_project(first_id)
                proj = self.project_manager.get_project(first_id)
                if proj:
                    self._load_project(proj)
        else:
            # 恢复选中原始项目
            real_proj = self.project_manager.get_project(real_id)
            if real_proj:
                self.project_list_widget.select_project(real_id)
                self._load_project(real_proj)

    def _cancel_stale_workers(self):
        """取消旧的后台线程（不阻塞），后续通过 generation 丢弃过时结果"""
        if self._scanner and self._scanner.isRunning():
            self._scanner.cancel()
        if self._thumbnail_loader and self._thumbnail_loader.isRunning():
            self._thumbnail_loader.cancel()
        self.preview_widget._cancel_preview_worker()

    def _shutdown_workers(self):
        """取消所有后台线程并等待它们结束，用于页面销毁前的清理。"""
        workers = [self._scanner, self._thumbnail_loader]
        pw = self.preview_widget._preview_worker
        if pw is not None:
            workers.append(pw)
        for w in workers:
            if w is not None and w.isRunning():
                w.cancel()
        for w in workers:
            if w is not None and w.isRunning():
                w.wait(3000)

    def closeEvent(self, event):
        self._shutdown_workers()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.current_project and not (self._scanner and self._scanner.isRunning()):
            self._on_refresh_project()

    def _load_project(self, project: DatasetProject):
        """加载项目"""
        self._cancel_stale_workers()
        self._scan_generation += 1

        self.current_project = project
        self.image_infos.clear()
        self.image_paths.clear()
        self.filtered_image_paths.clear()
        self.image_list_panel.clear()
        self.preview_widget.set_image(None)
        self.statistics_panel.clear()

        # 归档聚合条目 -> 多目录扫描
        if project.is_archive_root and project.archive_id:
            dirs = self.project_manager.get_archive_directories(project.archive_id)
            if not dirs:
                self.status_label.setText("归档内没有有效的目录")
                return
            self.subfolder_switch.setVisible(False)
            self.subfolder_switch.setEnabled(False)
            self.refresh_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText(f"正在扫描归档: {project.name}...")
            gen = self._scan_generation
            self._scanner = ImageScanner(directories=dirs)
            self._scanner.progress.connect(self._on_scan_progress)
            self._scanner.finished.connect(
                lambda infos, stats, _g=gen: self._on_scan_finished(infos, stats, _g)
            )
            self._scanner.start()
            return

        if not os.path.isdir(project.directory):
            InfoBar.error(
                title="错误",
                content=f"目录不存在: {project.directory}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self,
            )
            self.status_label.setText("目录不存在")
            return

        # 更新子文件夹模式开关的可见性
        real_id = project.parent_id if project.is_virtual else project.id
        if not project.archive_id:
            has_subs = self.project_manager.has_subfolders(real_id)
            self.subfolder_switch.setVisible(has_subs)
            self.subfolder_switch.setEnabled(has_subs)
            self.subfolder_switch.blockSignals(True)
            self.subfolder_switch.setChecked(self.project_manager.is_subfolder_mode(real_id))
            self.subfolder_switch.blockSignals(False)
        else:
            self.subfolder_switch.setVisible(False)
            self.subfolder_switch.setEnabled(False)

        self.refresh_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"正在扫描: {project.name}...")

        recursive = not (project.is_virtual and project.subfolder == "(root)")
        gen = self._scan_generation
        self._scanner = ImageScanner(project.directory, recursive=recursive)
        self._scanner.progress.connect(self._on_scan_progress)
        self._scanner.finished.connect(
            lambda infos, stats, _g=gen: self._on_scan_finished(infos, stats, _g)
        )
        self._scanner.start()
    
    def _on_scan_progress(self, current: int, total: int):
        """扫描进度"""
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.status_label.setText(f"正在扫描: {current}/{total}")
    
    def _on_scan_finished(self, image_infos: List[ImageInfo], stats: AnnotationStats,
                          generation: int = -1):
        """扫描完成"""
        if generation != self._scan_generation:
            return
        self.image_infos = sorted(image_infos, key=lambda i: i.path)
        self.image_paths = [info.path for info in self.image_infos]
        count = len(self.image_paths)

        self.statistics_panel.set_stats(stats)

        if self.current_project:
            self.current_project.image_count = count
            self.current_project.annotated_count = stats.annotated_images
            if not self.current_project.is_virtual:
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
        
        # 更新筛选项与图片列表（set_images -> _show_page -> page_changed -> _load_page_thumbnails）
        image_types = sorted({info.image_type for info in self.image_infos if info.image_type})
        self.image_list_panel.set_type_options(image_types)
        self._apply_filters()
    
    def _load_page_thumbnails(self, page_paths: List[str]):
        """加载当前页的缩略图"""
        if self._thumbnail_loader and self._thumbnail_loader.isRunning():
            self._thumbnail_loader.cancel()
        self._thumb_generation += 1
        
        uncached = self.image_list_panel.get_uncached_paths(page_paths)
        if not uncached:
            self.progress_bar.setVisible(False)
            self._update_status_after_load()
            return
        
        self._page_thumb_total = len(uncached)
        self._page_thumb_loaded = 0
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        gen = self._thumb_generation
        self._thumbnail_loader = ThumbnailLoader(uncached)
        self._thumbnail_loader.thumbnail_loaded.connect(
            lambda path, img, _g=gen: self._on_thumbnail_loaded(path, img, _g)
        )
        self._thumbnail_loader.all_loaded.connect(
            lambda _g=gen: self._on_page_thumbnails_loaded(_g)
        )
        self._thumbnail_loader.start()
    
    def _on_thumbnail_loaded(self, path: str, image: QImage, generation: int = -1):
        """缩略图加载 - 在主线程中将 QImage 转换为 QPixmap"""
        if generation != self._thumb_generation:
            return
        pixmap = QPixmap.fromImage(image)
        self.image_list_panel.update_thumbnail(path, pixmap)
        
        self._page_thumb_loaded += 1
        if self._page_thumb_total > 0:
            self.progress_bar.setValue(int(self._page_thumb_loaded / self._page_thumb_total * 100))
    
    def _on_page_thumbnails_loaded(self, generation: int = -1):
        """当前页缩略图加载完成"""
        if generation != self._thumb_generation:
            return
        self.progress_bar.setVisible(False)
        self._update_status_after_load()
    
    def _update_status_after_load(self):
        """缩略图加载后更新状态栏"""
        if self.current_project:
            panel = self.image_list_panel
            total = len(self.filtered_image_paths)
            page = panel._current_page + 1
            pages = panel._total_pages
            self.status_label.setText(
                f"{self.current_project.name}: 筛选后 {total} 张，第 {page}/{pages} 页"
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
        self.batch_annotate_btn.setEnabled(False)
    
    def _on_image_selected(self, image_path: str):
        """图片选择"""
        if self.current_project:
            self.preview_widget.project_directory = self.current_project.directory
        self.preview_widget.set_image(image_path)
        self.annotate_btn.setEnabled(True)
        selected_count = len(self.image_list_panel.get_selected_paths())
        self.batch_annotate_btn.setEnabled(selected_count >= 2)
    
    def _on_image_double_clicked(self, image_path: str):
        """图片双击"""
        self._open_for_annotation(image_path)
    
    def _on_annotate_clicked(self):
        """打开标注"""
        path = self.image_list_panel.get_selected_path()
        if path:
            self._open_for_annotation(path)
    
    def _on_batch_annotate_clicked(self):
        """打开批量标注"""
        paths = self.image_list_panel.get_selected_paths()
        if len(paths) < 2:
            InfoBar.warning(
                title="提示",
                content="请至少选择2张图片进行批量标注",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self,
            )
            return
        directory = ""
        if self.current_project:
            directory = self.current_project.directory
        else:
            directory = str(Path(paths[0]).parent)
        self.request_batch_annotation.emit(directory, paths)
    
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
