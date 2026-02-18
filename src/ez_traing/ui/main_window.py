import os

from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import FluentWindow, NavigationItemPosition

from ez_traing.pages.annotation_page import AnnotationPage
from ez_traing.pages.data_prep_page import DataPrepPage
from ez_traing.pages.dataset_page import DatasetPage
from ez_traing.pages.eval_page import EvalPage
from ez_traing.pages.prelabeling_page import PrelabelingPage
from ez_traing.pages.settings_page import SettingsPage
from ez_traing.pages.train_page import TrainPage


class LazyPageHost(QWidget):
    """延迟创建页面容器，首次展示时再初始化真实页面。"""

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory
        self._page = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def ensure_page(self):
        if self._page is None:
            self._page = self._factory()
            self._layout.addWidget(self._page)
        return self._page

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_page()


class AppWindow(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FluentLabel")
        self.resize(1280, 800)

        self.dataset_page = DatasetPage(self)
        self.data_prep_page = DataPrepPage(self)
        self.prelabeling_page = PrelabelingPage(self)
        self._annotation_page = None
        self._train_page = None
        self._eval_page = None
        self._settings_page = None
        self.annotation_page = LazyPageHost(self._create_annotation_page, self)
        self.train_page = LazyPageHost(self._create_train_page, self)
        self.eval_page = LazyPageHost(self._create_eval_page, self)
        self.settings_page = LazyPageHost(self._create_settings_page, self)

        self.annotation_page.setObjectName("annotation")
        self.dataset_page.setObjectName("dataset")
        self.data_prep_page.setObjectName("data_prep")
        self.train_page.setObjectName("train")
        self.prelabeling_page.setObjectName("prelabeling")
        self.eval_page.setObjectName("eval")
        self.settings_page.setObjectName("settings")

        self.addSubInterface(self.dataset_page, FIF.FOLDER, "数据集")
        self.addSubInterface(self.prelabeling_page, FIF.TAG, "预标注")
        self.addSubInterface(self.annotation_page, FIF.PHOTO, "标注")
        self.addSubInterface(self.data_prep_page, FIF.DOCUMENT, "数据准备")
        self.addSubInterface(self.train_page, FIF.ROBOT, "训练")
        self.addSubInterface(self.eval_page, FIF.COMPLETED, "验证")
        self.addSubInterface(
            self.settings_page,
            FIF.SETTING,
            "设置",
            NavigationItemPosition.BOTTOM,
        )

        # 共享 ProjectManager 给预标注页面
        self.prelabeling_page.set_project_manager(self.dataset_page.project_manager)
        # 共享 ProjectManager 给数据准备页面
        self.data_prep_page.set_project_manager(self.dataset_page.project_manager)

        # 连接数据集页面的标注联动信号
        self.dataset_page.request_annotation.connect(self._on_request_annotation)

    def _create_annotation_page(self):
        if self._annotation_page is None:
            self._annotation_page = AnnotationPage(self)
        return self._annotation_page

    def _create_train_page(self):
        if self._train_page is None:
            self._train_page = TrainPage(self)
        return self._train_page

    def _create_eval_page(self):
        if self._eval_page is None:
            self._eval_page = EvalPage(self)
        return self._eval_page

    def _create_settings_page(self):
        if self._settings_page is None:
            self._settings_page = SettingsPage(self)
        return self._settings_page

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

        # 切换到标注页面
        self.switchTo(self.annotation_page)

        # 打开项目文件夹和图片进行标注
        annotation_window = self._annotation_window()
        if annotation_window:
            # 先导入整个目录的图片
            if directory and os.path.isdir(directory):
                # 检查是否需要重新加载目录（如果目录已经加载，避免重复加载）
                current_dir = getattr(annotation_window, "dir_name", None)
                if current_dir != directory:
                    annotation_window.import_dir_images(directory)
                    annotation_window.default_save_dir = directory
            
            # 然后加载指定的图片（会自动在列表中高亮定位）
            annotation_window.load_file(image_path)


