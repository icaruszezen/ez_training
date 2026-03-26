from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QAbstractSpinBox, QSpinBox


class ZoomWidget(QSpinBox):

    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setRange(10, 500)
        self.setSuffix(' %')
        self.setValue(value)
        self.setToolTip(u'Zoom Level')
        self.setStatusTip(self.toolTip())
        self.setAlignment(Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(ZoomWidget, self).minimumSizeHint().height()
        fm = QFontMetrics(self.font())
        width = fm.horizontalAdvance(str(self.maximum()))
        return QSize(width, height)
