from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt, QPoint, QStringListModel
from PyQt5.QtWidgets import (
    QCompleter, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QPushButton, QVBoxLayout,
)

from ez_training.labeling.utils import label_validator, trimmed

BB = QDialogButtonBox

FLUENT_DIALOG_STYLE = """
LabelDialog {
    background-color: #FCFCFC;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
}
LabelDialog QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #D1D1D1;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    color: #1A1A1A;
    selection-background-color: #0078D4;
    selection-color: #FFFFFF;
}
LabelDialog QLineEdit:focus {
    border: 1px solid #0078D4;
    border-bottom: 2px solid #0078D4;
}
LabelDialog QLineEdit:hover:!focus {
    border-color: #A0A0A0;
}
LabelDialog QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}
LabelDialog QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    color: #1A1A1A;
    margin: 2px 4px;
}
LabelDialog QListWidget::item:hover {
    background-color: #F5F5F5;
}
LabelDialog QListWidget::item:selected {
    background-color: #E5F1FB;
    color: #1A1A1A;
}
LabelDialog QListWidget::item:selected:hover {
    background-color: #CCE4F7;
}
LabelDialog QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #D1D1D1;
    border-radius: 4px;
    padding: 6px 16px;
    font-size: 13px;
    color: #1A1A1A;
    min-width: 70px;
}
LabelDialog QPushButton:hover {
    background-color: #F5F5F5;
    border-color: #C1C1C1;
}
LabelDialog QPushButton:pressed {
    background-color: #E5E5E5;
    border-color: #A0A0A0;
}
LabelDialog QPushButton:default, LabelDialog QPushButton[default="true"] {
    background-color: #0078D4;
    border: 1px solid #0067B8;
    color: #FFFFFF;
}
LabelDialog QPushButton:default:hover, LabelDialog QPushButton[default="true"]:hover {
    background-color: #1084D9;
    border-color: #0078D4;
}
LabelDialog QPushButton:default:pressed, LabelDialog QPushButton[default="true"]:pressed {
    background-color: #006CBD;
    border-color: #005A9E;
}
LabelDialog QDialogButtonBox {
    dialogbuttonbox-buttons-have-icons: 0;
}
"""


class LabelDialog(QDialog):

    def __init__(self, text="Enter object label", parent=None, list_item=None):
        super(LabelDialog, self).__init__(parent)

        self.setStyleSheet(FLUENT_DIALOG_STYLE)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel("选择标签")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #1A1A1A;
                padding-bottom: 4px;
            }
        """)
        layout.addWidget(title_label)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("输入或搜索标签...")
        self.edit.setText(text if text != "Enter object label" else "")
        self.edit.setValidator(label_validator())
        self.edit.editingFinished.connect(self.post_process)
        self.edit.setMinimumHeight(36)
        layout.addWidget(self.edit)

        if list_item:
            model = QStringListModel()
            model.setStringList(list_item)
            completer = QCompleter()
            completer.setModel(model)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            self.edit.setCompleter(completer)

        if list_item is not None and len(list_item) > 0:
            self.list_widget = QListWidget(self)
            self.list_widget.setMinimumHeight(150)
            self.list_widget.setMaximumHeight(250)
            for item in list_item:
                self.list_widget.addItem(item)
            self.list_widget.itemClicked.connect(self.list_item_click)
            self.list_widget.itemDoubleClicked.connect(self.list_item_double_click)
            layout.addWidget(self.list_widget)

            self.edit.textChanged.connect(self._filter_list)
            self._all_items = list_item

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton("确定")
        self.ok_btn.setDefault(True)
        self.ok_btn.setProperty("default", True)
        self.ok_btn.clicked.connect(self.validate)
        btn_layout.addWidget(self.ok_btn)

        layout.addLayout(btn_layout)

        self.button_box = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        self.button_box.hide()
        self.button_box.accepted.connect(self.validate)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(layout)
        self.setMinimumWidth(320)

    def _filter_list(self, text):
        if not hasattr(self, 'list_widget') or not hasattr(self, '_all_items'):
            return
        search_text = text.lower()
        self.list_widget.clear()
        for item in self._all_items:
            if search_text in item.lower():
                self.list_widget.addItem(item)

    def validate(self):
        if trimmed(self.edit.text()):
            self.accept()

    def post_process(self):
        self.edit.setText(trimmed(self.edit.text()))

    def pop_up(self, text='', move=True):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)

        if hasattr(self, 'list_widget') and hasattr(self, '_all_items'):
            self.list_widget.clear()
            for item in self._all_items:
                self.list_widget.addItem(item)

        if move:
            cursor_pos = QCursor.pos()
            self.adjustSize()

            btn = self.ok_btn
            btn.adjustSize()
            offset = btn.mapToGlobal(btn.pos()) - self.pos()
            offset += QPoint(btn.size().width() // 4, btn.size().height() // 2)
            cursor_pos.setX(max(0, cursor_pos.x() - offset.x()))
            cursor_pos.setY(max(0, cursor_pos.y() - offset.y()))

            parent = self.parentWidget()
            if parent:
                parent_bottom_right = parent.geometry()
                max_x = parent_bottom_right.x() + parent_bottom_right.width() - self.sizeHint().width()
                max_y = parent_bottom_right.y() + parent_bottom_right.height() - self.sizeHint().height()
                max_global = parent.mapToGlobal(QPoint(max_x, max_y))
                if cursor_pos.x() > max_global.x():
                    cursor_pos.setX(max_global.x())
                if cursor_pos.y() > max_global.y():
                    cursor_pos.setY(max_global.y())
            self.move(cursor_pos)
        return trimmed(self.edit.text()) if self.exec_() else None

    def list_item_click(self, t_qlist_widget_item):
        text = trimmed(t_qlist_widget_item.text())
        self.edit.setText(text)

    def list_item_double_click(self, t_qlist_widget_item):
        self.list_item_click(t_qlist_widget_item)
        self.validate()
