from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    def __init__(self, title, description, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title, self)
        title_label.setObjectName("placeholderTitle")
        title_label.setAlignment(Qt.AlignCenter)

        desc_label = QLabel(description, self)
        desc_label.setObjectName("placeholderDesc")
        desc_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
