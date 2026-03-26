from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ez_training.labeling.annotation_window import AnnotationWindow


class AnnotationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.annotation_window = AnnotationWindow(parent=self)
        layout.addWidget(self.annotation_window)
