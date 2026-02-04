import os
import re
import sys
import types
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPalette

LABELIMG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "labelImg"
RESOURCES_ROOT = LABELIMG_ROOT / "resources"
ICONS_DIR = RESOURCES_ROOT / "icons"
STRINGS_DIR = RESOURCES_ROOT / "strings"


def _ensure_labelimg_path():
    if str(LABELIMG_ROOT) not in sys.path:
        sys.path.insert(0, str(LABELIMG_ROOT))


def _ensure_resources_stub():
    if "libs.resources" not in sys.modules:
        sys.modules["libs.resources"] = types.ModuleType("libs.resources")


def _resolve_icon_path(icon_name):
    alias_map = {
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

    filename = alias_map.get(icon_name)
    if filename:
        candidate = ICONS_DIR / filename
        if candidate.exists():
            return candidate

    for ext in (".png", ".svg", ".ico"):
        candidate = ICONS_DIR / f"{icon_name}{ext}"
        if candidate.exists():
            return candidate

    return None


def _new_icon(icon_name):
    path = _resolve_icon_path(icon_name)
    return QIcon(str(path)) if path else QIcon()


def _apply_fluent_palette(widget):
    palette = widget.palette()
    palette.setColor(QPalette.Window, QColor("#F3F4F7"))
    palette.setColor(QPalette.WindowText, QColor("#1F1F1F"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.AlternateBase, QColor("#F6F6F8"))
    palette.setColor(QPalette.Text, QColor("#1F1F1F"))
    palette.setColor(QPalette.Button, QColor("#F7F7F9"))
    palette.setColor(QPalette.ButtonText, QColor("#1F1F1F"))
    palette.setColor(QPalette.Highlight, QColor("#DDEBFF"))
    palette.setColor(QPalette.HighlightedText, QColor("#1F1F1F"))
    palette.setColor(QPalette.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipText, QColor("#1F1F1F"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#9E9E9E"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9E9E9E"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#9E9E9E"))
    widget.setPalette(palette)


def _apply_fluent_stylesheet(widget):
    accent = "#0078D4"
    widget.setObjectName("fluentAnnotationWindow")
    widget.setAttribute(Qt.WA_StyledBackground, True)
    widget.setStyleSheet(
        f"""
#fluentAnnotationWindow {{
    background-color: #F3F4F7;
}}
#fluentAnnotationWindow QMenuBar {{
    background-color: #F7F7F9;
    color: #1F1F1F;
}}
#fluentAnnotationWindow QMenuBar::item:selected {{
    background-color: #E9EEF6;
}}
#fluentAnnotationWindow QMenu {{
    background-color: #FFFFFF;
    color: #1F1F1F;
    border: 1px solid #E3E3E3;
}}
#fluentAnnotationWindow QMenu::item:selected {{
    background-color: #E9EEF6;
}}
#fluentAnnotationWindow QToolBar {{
    background-color: #F7F7F9;
    border: 1px solid #E3E3E3;
}}
#fluentAnnotationWindow QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    color: #1F1F1F;
    border-radius: 6px;
}}
#fluentAnnotationWindow QToolButton:hover {{
    background-color: #E9EEF6;
    border-color: #D6E4FF;
}}
#fluentAnnotationWindow QToolButton:pressed {{
    background-color: #DDEBFF;
    border-color: #C4D8FF;
}}
#fluentAnnotationWindow QToolButton:checked {{
    background-color: #DDEBFF;
    border-color: #B7D0FF;
}}
#fluentAnnotationWindow QDockWidget {{
    background-color: #F7F7F9;
    border: 1px solid #E3E3E3;
}}
#fluentAnnotationWindow QDockWidget::title {{
    background-color: #F0F2F6;
    color: #1F1F1F;
}}
#fluentAnnotationWindow QListWidget,
#fluentAnnotationWindow QTreeView,
#fluentAnnotationWindow QTableView {{
    background-color: #FFFFFF;
    border: 1px solid #E3E3E3;
    color: #1F1F1F;
}}
#fluentAnnotationWindow QListWidget::item:selected,
#fluentAnnotationWindow QTreeView::item:selected,
#fluentAnnotationWindow QTableView::item:selected {{
    background-color: #DDEBFF;
    color: #1F1F1F;
}}
#fluentAnnotationWindow QLineEdit,
#fluentAnnotationWindow QTextEdit,
#fluentAnnotationWindow QPlainTextEdit,
#fluentAnnotationWindow QComboBox,
#fluentAnnotationWindow QSpinBox,
#fluentAnnotationWindow QDoubleSpinBox {{
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 4px;
    color: #1F1F1F;
}}
#fluentAnnotationWindow QLineEdit:focus,
#fluentAnnotationWindow QTextEdit:focus,
#fluentAnnotationWindow QPlainTextEdit:focus,
#fluentAnnotationWindow QComboBox:focus,
#fluentAnnotationWindow QSpinBox:focus,
#fluentAnnotationWindow QDoubleSpinBox:focus {{
    border-color: {accent};
}}
#fluentAnnotationWindow QCheckBox::indicator,
#fluentAnnotationWindow QRadioButton::indicator {{
    background-color: #FFFFFF;
    border: 1px solid #8A8A8A;
    border-radius: 3px;
}}
#fluentAnnotationWindow QCheckBox::indicator:checked,
#fluentAnnotationWindow QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}
#fluentAnnotationWindow QStatusBar {{
    background-color: #F7F7F9;
    color: #1F1F1F;
}}
#fluentAnnotationWindow QToolTip {{
    background-color: #FFFFFF;
    color: #1F1F1F;
    border: 1px solid #E3E3E3;
}}
"""
    )


def _apply_fluent_style(widget):
    _apply_fluent_palette(widget)
    _apply_fluent_stylesheet(widget)


def _patch_string_bundle(label_string_bundle):
    def _create_lookup_fallback_list(self, locale_str):
        result_paths = []
        base_path = STRINGS_DIR / "strings"
        result_paths.append(str(base_path))
        if locale_str is not None:
            tags = re.split("[^a-zA-Z]", locale_str)
            for tag in tags:
                last_path = result_paths[-1]
                result_paths.append(last_path + "-" + tag)
        return result_paths

    def _load_bundle(self, path):
        filename = f"{path}.properties"
        if not os.path.exists(filename):
            return
        with open(filename, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                key_value = line.split("=")
                key = key_value[0].strip()
                value = "=".join(key_value[1:]).strip().strip('"')
                self.id_to_message[key] = value

    label_string_bundle.StringBundle._StringBundle__create_lookup_fallback_list = _create_lookup_fallback_list
    label_string_bundle.StringBundle._StringBundle__load_bundle = _load_bundle


def _patch_utils(label_utils):
    label_utils.new_icon = _new_icon


def _apply_labelimg_patches():
    _ensure_labelimg_path()
    _ensure_resources_stub()

    from libs import stringBundle as label_string_bundle
    from libs import utils as label_utils

    if "libs" in sys.modules and "libs.resources" in sys.modules:
        setattr(sys.modules["libs"], "resources", sys.modules["libs.resources"])

    _patch_string_bundle(label_string_bundle)
    _patch_utils(label_utils)


_apply_labelimg_patches()

import labelImg as labelimg_module

labelimg_module.__appname__ = "FluentLabel"


class AnnotationWindow(labelimg_module.MainWindow):
    def __init__(
        self,
        default_filename=None,
        default_prefdef_class_file=None,
        default_save_dir=None,
        parent=None,
    ):
        if default_prefdef_class_file is None:
            default_prefdef_class_file = str(LABELIMG_ROOT / "data" / "predefined_classes.txt")

        super().__init__(default_filename, default_prefdef_class_file, default_save_dir)

        if parent is not None:
            self.setParent(parent)
            self.setWindowFlags(Qt.Widget)

        _apply_fluent_style(self)
