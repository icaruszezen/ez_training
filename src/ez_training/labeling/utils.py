import hashlib
import re
import sys
from math import sqrt
from pathlib import Path

from PyQt5.QtGui import QColor, QIcon, QRegExpValidator
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import QAction, QMenu, QPushButton

from ez_training.labeling.ustr import ustr

if getattr(sys, "frozen", False):
    _LABELING_ROOT = Path(sys._MEIPASS) / "ez_training" / "labeling"
else:
    _LABELING_ROOT = Path(__file__).resolve().parent

ICONS_DIR = _LABELING_ROOT / "resources" / "icons"

_ICON_ALIAS_MAP = {
    "help": "help.png",
    "app": "app.png",
    "expert": "expert2.png",
    "done": "done.png",
    "file": "file.png",
    "labels": "labels.png",
    "new": "objects.png",
    "close": "close.png",
    "fit-width": "fit-width.png",
    "fit-window": "fit-window.png",
    "undo": "undo.png",
    "hide": "eye.png",
    "quit": "quit.png",
    "copy": "copy.png",
    "edit": "edit.png",
    "open": "open.png",
    "save": "save.png",
    "format_voc": "format_voc.png",
    "format_yolo": "format_yolo.png",
    "format_createml": "format_createml.png",
    "save-as": "save-as.png",
    "color": "color.png",
    "color_line": "color_line.png",
    "zoom": "zoom.png",
    "zoom-in": "zoom-in.png",
    "zoom-out": "zoom-out.png",
    "light_reset": "light_reset.png",
    "light_lighten": "light_lighten.png",
    "light_darken": "light_darken.png",
    "delete": "cancel.png",
    "next": "next.png",
    "prev": "prev.png",
    "resetall": "resetall.png",
    "verify": "verify.png",
}


def _resolve_icon_path(icon_name):
    filename = _ICON_ALIAS_MAP.get(icon_name)
    if filename:
        candidate = ICONS_DIR / filename
        if candidate.exists():
            return candidate

    for ext in (".png", ".svg", ".ico"):
        candidate = ICONS_DIR / f"{icon_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def new_icon(icon):
    path = _resolve_icon_path(icon)
    return QIcon(str(path)) if path else QIcon()


def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def format_shortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def generate_color_by_text(text):
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, 100)


def have_qstring():
    return False


def util_qt_strlistclass():
    return list


def natural_sort(list, key=lambda s: s):
    """Sort the list into natural alphanumeric order."""
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)


def trimmed(text):
    return text.strip()
