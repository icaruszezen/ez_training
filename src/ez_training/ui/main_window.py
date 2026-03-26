import logging
import os

from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import FluentWindow, InfoBar, NavigationItemPosition

from ez_training.pages.annotation_guide_page import AnnotationGuidePage
from ez_training.pages.annotation_page import AnnotationPage
from ez_training.pages.batch_annotation_page import BatchAnnotationPage
from ez_training.pages.data_prep_page import DataPrepPage
from ez_training.pages.dataset_page import DatasetPage
from ez_training.pages.eval_page import EvalPage
from ez_training.pages.prelabeling_page import PrelabelingPage
from ez_training.pages.script_annotation_page import ScriptAnnotationPage
from ez_training.pages.settings_page import SettingsPage
from ez_training.pages.template_matching_page import TemplateMatchingPage
from ez_training.pages.tools_page import ToolsPage
from ez_training.pages.train_page import TrainPage

logger = logging.getLogger(__name__)


class LazyPageHost(QWidget):
    """延迟创建页面容器，首次展示时再初始化真实页面。仅限 GUI 线程调用。"""

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory
        self._page = None
        self._failed = False
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def ensure_page(self):
        if self._page is None and not self._failed:
            try:
                self._page = self._factory()
                self._layout.addWidget(self._page)
            except Exception:
                self._failed = True
                logger.exception("Failed to create page")
                InfoBar.error(
                    "页面加载失败",
                    "无法初始化此功能页面，请检查日志",
                    parent=self,
                    duration=5000,
                )
        return self._page

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_page()


class AppWindow(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EZ Training")
        self.resize(1280, 800)

        self.dataset_page = DatasetPage(self)
        self.data_prep_page = DataPrepPage(self)
        self.prelabeling_page = PrelabelingPage(self)
        self.annotation_page = LazyPageHost(self._create_annotation_page, self)
        self.batch_annotation_page = LazyPageHost(self._create_batch_annotation_page, self)
        self.template_matching_page = LazyPageHost(self._create_template_matching_page, self)
        self.script_annotation_page = LazyPageHost(self._create_script_annotation_page, self)
        self.train_page = LazyPageHost(self._create_train_page, self)
        self.eval_page = LazyPageHost(self._create_eval_page, self)
        self.tools_page = LazyPageHost(self._create_tools_page, self)
        self.annotation_guide_page = LazyPageHost(self._create_annotation_guide_page, self)
        self.settings_page = LazyPageHost(self._create_settings_page, self)

        self.annotation_page.setObjectName("annotation")
        self.batch_annotation_page.setObjectName("batch_annotation")
        self.dataset_page.setObjectName("dataset")
        self.data_prep_page.setObjectName("data_prep")
        self.template_matching_page.setObjectName("template_matching")
        self.script_annotation_page.setObjectName("script_annotation")
        self.train_page.setObjectName("train")
        self.prelabeling_page.setObjectName("prelabeling")
        self.eval_page.setObjectName("eval")
        self.tools_page.setObjectName("tools")
        self.annotation_guide_page.setObjectName("annotation_guide")
        self.settings_page.setObjectName("settings")

        self.addSubInterface(self.dataset_page, FIF.FOLDER, "数据集")
        self.addSubInterface(self.prelabeling_page, FIF.TAG, "预标注")
        self.addSubInterface(self.annotation_page, FIF.PHOTO, "标注")
        self.addSubInterface(self.batch_annotation_page, FIF.COPY, "批量标注")
        self.addSubInterface(self.template_matching_page, FIF.SEARCH, "模板匹配")
        self.addSubInterface(self.data_prep_page, FIF.DOCUMENT, "数据准备")
        self.addSubInterface(self.script_annotation_page, FIF.CODE, "脚本标注")
        self.addSubInterface(self.train_page, FIF.ROBOT, "训练")
        self.addSubInterface(self.eval_page, FIF.COMPLETED, "验证")
        self.addSubInterface(self.tools_page, FIF.DEVELOPER_TOOLS, "小工具")
        self.addSubInterface(self.annotation_guide_page, FIF.EDIT, "标注指导")
        self.addSubInterface(
            self.settings_page,
            FIF.SETTING,
            "设置",
            NavigationItemPosition.BOTTOM,
        )

        # 共享 ProjectManager
        self.prelabeling_page.set_project_manager(self.dataset_page.project_manager)
        self.data_prep_page.set_project_manager(self.dataset_page.project_manager)

        # 连接数据集页面的标注联动信号
        self.dataset_page.request_annotation.connect(self._on_request_annotation)
        self.dataset_page.request_batch_annotation.connect(self._on_request_batch_annotation)

    def _create_annotation_page(self):
        return AnnotationPage(self)

    def _create_batch_annotation_page(self):
        page = BatchAnnotationPage(self)
        page.set_project_manager(self.dataset_page.project_manager)
        return page

    def _create_template_matching_page(self):
        page = TemplateMatchingPage(self)
        page.set_project_manager(self.dataset_page.project_manager)
        return page

    def _create_script_annotation_page(self):
        page = ScriptAnnotationPage(self)
        page.set_project_manager(self.dataset_page.project_manager)
        return page

    def _create_train_page(self):
        return TrainPage(self)

    def _create_eval_page(self):
        page = EvalPage(self)
        page.set_project_manager(self.dataset_page.project_manager)
        return page

    def _create_tools_page(self):
        return ToolsPage(self)

    def _create_annotation_guide_page(self):
        page = AnnotationGuidePage(self)
        page.set_project_manager(self.dataset_page.project_manager)
        return page

    def _create_settings_page(self):
        return SettingsPage(self)

    def _annotation_window(self):
        annotation_page = self.annotation_page.ensure_page()
        return getattr(annotation_page, "annotation_window", None)

    @property
    def file_path(self):
        annotation_window = self._annotation_window()
        return annotation_window.file_path if annotation_window else None

    @property
    def label_coordinates(self):
        annotation_window = self._annotation_window()
        return getattr(annotation_window, "label_coordinates", None)

    def _on_request_annotation(self, directory: str, image_path: str):
        """处理数据集页面的标注请求，跳转到标注页面并打开项目文件夹和图片"""
        if not image_path or not os.path.exists(image_path):
            return

        self.switchTo(self.annotation_page)

        annotation_window = self._annotation_window()
        if annotation_window:
            if directory and os.path.isdir(directory):
                current_dir = getattr(annotation_window, "dir_name", None)
                if current_dir != directory:
                    annotation_window.import_dir_images(directory)
                    annotation_window.default_save_dir = directory
            annotation_window.load_file(image_path)

    def _on_request_batch_annotation(self, directory: str, image_paths: list):
        """处理数据集页面的批量标注请求"""
        if not image_paths:
            return

        self.switchTo(self.batch_annotation_page)

        page = self.batch_annotation_page.ensure_page()
        if page:
            page.load_images(directory, image_paths)


