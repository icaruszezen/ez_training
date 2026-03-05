"""Python 标注脚本管理页面。"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget,
    CaptionLabel,
    ComboBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SubtitleLabel,
    TextEdit,
    TitleLabel,
)


class ScriptAnnotationPage(QWidget):
    """标注脚本管理页面。"""

    _SCRIPT_TEMPLATE = '''"""自定义标注脚本模板。"""

import argparse


def run(dataset_dir: str) -> None:
    print(f"开始处理数据集: {dataset_dir}")
    # TODO: 在此处编写你的标注逻辑
    print("处理完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="自定义标注脚本")
    parser.add_argument("--dataset_dir", required=True, help="数据集目录")
    args = parser.parse_args()
    run(args.dataset_dir)


if __name__ == "__main__":
    main()
'''

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = None
        self._project_ids: List[str] = []
        self._current_project_id: Optional[str] = None

        self._script_paths: List[Path] = []
        self._current_script_path: Optional[Path] = None
        self._is_dirty = False

        self._process: Optional[QProcess] = None

        self._setup_ui()
        self._ensure_script_dir()
        self._refresh_script_list()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = ScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(36, 20, 36, 20)
        content_layout.setSpacing(16)

        content_layout.addWidget(TitleLabel("脚本标注", self))
        content_layout.addWidget(self._create_script_manage_card())
        content_layout.addWidget(self._create_editor_card())
        content_layout.addWidget(self._create_run_card())
        content_layout.addWidget(self._create_log_card())
        content_layout.addStretch()

        scroll_area.setWidget(content)
        main_layout.addWidget(scroll_area)

    def _create_script_manage_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("脚本管理", card))

        name_row = QHBoxLayout()
        self.script_name_edit = LineEdit(card)
        self.script_name_edit.setPlaceholderText("输入脚本名称（无需 .py）")
        name_row.addWidget(self.script_name_edit)

        self.new_script_btn = PushButton("新建", card)
        self.new_script_btn.setIcon(FIF.ADD)
        self.new_script_btn.clicked.connect(self._on_new_script)
        name_row.addWidget(self.new_script_btn)

        self.refresh_script_btn = PushButton("刷新", card)
        self.refresh_script_btn.setIcon(FIF.SYNC)
        self.refresh_script_btn.clicked.connect(self._refresh_script_list)
        name_row.addWidget(self.refresh_script_btn)

        self.delete_script_btn = PushButton("删除", card)
        self.delete_script_btn.setIcon(FIF.DELETE)
        self.delete_script_btn.clicked.connect(self._on_delete_script)
        name_row.addWidget(self.delete_script_btn)

        layout.addLayout(name_row)

        self.script_dir_label = CaptionLabel("", card)
        self.script_dir_label.setWordWrap(True)
        layout.addWidget(self.script_dir_label)

        self.script_list = QListWidget(card)
        self.script_list.setMinimumHeight(160)
        self.script_list.currentRowChanged.connect(self._on_script_selected)
        layout.addWidget(self.script_list)

        return card

    def _create_editor_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(SubtitleLabel("脚本编辑", card))
        header.addStretch()

        self.current_script_label = CaptionLabel("当前脚本：未选择", card)
        header.addWidget(self.current_script_label)
        layout.addLayout(header)

        self.script_editor = TextEdit(card)
        self.script_editor.setFont(QFont("Consolas", 10))
        self.script_editor.setMinimumHeight(300)
        self.script_editor.textChanged.connect(self._on_editor_text_changed)
        layout.addWidget(self.script_editor)

        btn_row = QHBoxLayout()
        self.save_script_btn = PrimaryPushButton("保存脚本", card)
        self.save_script_btn.setIcon(FIF.SAVE)
        self.save_script_btn.clicked.connect(self._on_save_script)
        btn_row.addWidget(self.save_script_btn)

        self.reload_script_btn = PushButton("重新加载", card)
        self.reload_script_btn.setIcon(FIF.SYNC)
        self.reload_script_btn.clicked.connect(self._on_reload_script)
        btn_row.addWidget(self.reload_script_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        return card

    def _create_run_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(SubtitleLabel("数据集与运行", card))

        self.dataset_combo = ComboBox(card)
        self.dataset_combo.setPlaceholderText("请先在“数据集”页面创建项目")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        layout.addWidget(self.dataset_combo)

        self.dataset_info_label = CaptionLabel("未选择数据集", card)
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)

        btn_row = QHBoxLayout()
        self.refresh_dataset_btn = PushButton("刷新数据集", card)
        self.refresh_dataset_btn.setIcon(FIF.SYNC)
        self.refresh_dataset_btn.clicked.connect(self._refresh_dataset_list)
        btn_row.addWidget(self.refresh_dataset_btn)

        self.run_btn = PrimaryPushButton("运行脚本", card)
        self.run_btn.setIcon(FIF.PLAY)
        self.run_btn.clicked.connect(self._on_run_script)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = PushButton("停止", card)
        self.stop_btn.setIcon(FIF.CLOSE)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_script)
        btn_row.addWidget(self.stop_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        return card

    def _create_log_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(SubtitleLabel("运行日志", card))
        header.addStretch()

        clear_btn = PushButton("清空", card)
        clear_btn.clicked.connect(self._clear_log)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        self.log_edit = TextEdit(card)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(220)
        self.log_edit.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_edit)

        return card

    def set_project_manager(self, manager) -> None:
        self._project_manager = manager

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_dataset_list()
        self._refresh_script_list()

    def _script_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "annotation_scripts"

    def _ensure_script_dir(self) -> None:
        script_dir = self._script_dir()
        script_dir.mkdir(parents=True, exist_ok=True)
        self.script_dir_label.setText(f"脚本目录: {script_dir}")

    def _refresh_script_list(self) -> None:
        self._ensure_script_dir()
        prev_path = self._current_script_path

        self.script_list.blockSignals(True)
        self.script_list.clear()
        self._script_paths.clear()

        script_paths = sorted(self._script_dir().glob("*.py"), key=lambda p: p.name.lower())
        for script_path in script_paths:
            item = QListWidgetItem(script_path.name)
            item.setData(Qt.UserRole, str(script_path))
            self.script_list.addItem(item)
            self._script_paths.append(script_path)

        self.script_list.blockSignals(False)

        if not self._script_paths:
            self._current_script_path = None
            self.current_script_label.setText("当前脚本：未选择")
            self.script_editor.clear()
            return

        if prev_path and prev_path in self._script_paths:
            idx = self._script_paths.index(prev_path)
            self.script_list.setCurrentRow(idx)
        elif self._current_script_path and self._current_script_path in self._script_paths:
            idx = self._script_paths.index(self._current_script_path)
            self.script_list.setCurrentRow(idx)
        else:
            self.script_list.setCurrentRow(0)

    def _on_new_script(self) -> None:
        raw_name = self.script_name_edit.text().strip()
        if not raw_name:
            raw_name = "new_annotation_script"

        safe_name = re.sub(r"[^0-9A-Za-z_\-]", "_", raw_name)
        safe_name = safe_name.strip("._")
        if not safe_name:
            safe_name = "new_annotation_script"

        script_dir = self._script_dir()
        target = script_dir / f"{safe_name}.py"
        idx = 1
        while target.exists():
            target = script_dir / f"{safe_name}_{idx}.py"
            idx += 1

        try:
            target.write_text(self._SCRIPT_TEMPLATE, encoding="utf-8")
        except OSError as e:
            self._show_error("新建脚本失败", str(e))
            return

        self._append_log(f"已新建脚本: {target.name}")
        self.script_name_edit.clear()
        self._refresh_script_list()

        if target in self._script_paths:
            self.script_list.setCurrentRow(self._script_paths.index(target))

        InfoBar.success(
            title="新建成功",
            content=f"已创建脚本 {target.name}",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2500,
        )

    def _on_delete_script(self) -> None:
        if self._current_script_path is None:
            self._show_warning("删除脚本", "请先选择一个脚本")
            return
        if self._process and self._process.state() != QProcess.NotRunning:
            self._show_warning("删除脚本", "脚本运行中，无法删除")
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除脚本 {self._current_script_path.name} 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._current_script_path.unlink(missing_ok=False)
        except OSError as e:
            self._show_error("删除失败", str(e))
            return

        self._append_log(f"已删除脚本: {self._current_script_path.name}")
        self._current_script_path = None
        self._is_dirty = False
        self._refresh_script_list()

    def _on_script_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._script_paths):
            self._current_script_path = None
            self.current_script_label.setText("当前脚本：未选择")
            self.script_editor.clear()
            return

        target_path = self._script_paths[row]
        if self._current_script_path == target_path:
            return

        if not self._confirm_discard_if_dirty():
            self.script_list.blockSignals(True)
            if self._current_script_path in self._script_paths:
                self.script_list.setCurrentRow(self._script_paths.index(self._current_script_path))
            self.script_list.blockSignals(False)
            return

        self._load_script(target_path)

    def _load_script(self, script_path: Path) -> None:
        try:
            content = script_path.read_text(encoding="utf-8")
        except OSError as e:
            self._show_error("读取脚本失败", str(e))
            return

        self.script_editor.blockSignals(True)
        self.script_editor.setPlainText(content)
        self.script_editor.blockSignals(False)

        self._current_script_path = script_path
        self._is_dirty = False
        self._update_current_script_label()
        self._append_log(f"已加载脚本: {script_path.name}")

    def _on_editor_text_changed(self) -> None:
        if self._current_script_path is None:
            return
        self._is_dirty = True
        self._update_current_script_label()

    def _update_current_script_label(self) -> None:
        if self._current_script_path is None:
            self.current_script_label.setText("当前脚本：未选择")
            return
        dirty_mark = " *" if self._is_dirty else ""
        self.current_script_label.setText(f"当前脚本：{self._current_script_path.name}{dirty_mark}")

    def _confirm_discard_if_dirty(self) -> bool:
        if not self._is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "未保存修改",
            "当前脚本有未保存修改，是否放弃修改并继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _on_save_script(self) -> None:
        if self._current_script_path is None:
            self._show_warning("保存脚本", "请先选择或新建脚本")
            return
        if self._process and self._process.state() != QProcess.NotRunning:
            self._show_warning("保存脚本", "脚本运行中，请先停止后再保存")
            return

        try:
            self._current_script_path.write_text(self.script_editor.toPlainText(), encoding="utf-8")
        except OSError as e:
            self._show_error("保存失败", str(e))
            return

        self._is_dirty = False
        self._update_current_script_label()
        self._append_log(f"已保存脚本: {self._current_script_path.name}")

        InfoBar.success(
            title="保存成功",
            content=f"已保存 {self._current_script_path.name}",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2000,
        )

    def _on_reload_script(self) -> None:
        if self._current_script_path is None:
            self._show_warning("重新加载", "请先选择一个脚本")
            return
        if not self._confirm_discard_if_dirty():
            return
        self._load_script(self._current_script_path)

    def _refresh_dataset_list(self) -> None:
        if self._project_manager is None:
            return

        prev_id = self._current_project_id
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self._project_ids.clear()

        projects = self._project_manager.get_all_projects(exclude_archived=True)
        for proj in projects:
            self.dataset_combo.addItem(f"{proj.name} ({proj.image_count} 张)")
            self._project_ids.append(proj.id)

        if not projects:
            self._current_project_id = None
            self.dataset_info_label.setText("请先在“数据集”页面创建项目")
        elif prev_id in self._project_ids:
            idx = self._project_ids.index(prev_id)
            self.dataset_combo.setCurrentIndex(idx)
            self._current_project_id = prev_id
            self._update_dataset_info(prev_id)
        else:
            self.dataset_combo.setCurrentIndex(0)
            self._current_project_id = self._project_ids[0]
            self._update_dataset_info(self._current_project_id)

        self.dataset_combo.blockSignals(False)

    def _on_dataset_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._project_ids):
            self._current_project_id = None
            self.dataset_info_label.setText("未选择数据集")
            return

        self._current_project_id = self._project_ids[index]
        self._update_dataset_info(self._current_project_id)

    def _update_dataset_info(self, project_id: str) -> None:
        if self._project_manager is None:
            return
        project = self._project_manager.get_project(project_id)
        if not project:
            self.dataset_info_label.setText("数据集不存在")
            return
        if project.is_archive_root:
            dirs = self._project_manager.get_archive_directories(project.archive_id)
            self.dataset_info_label.setText(
                f"归档: {project.name}\n包含 {len(dirs)} 个目录\n记录图片数: {project.image_count}"
            )
        else:
            self.dataset_info_label.setText(
                f"项目: {project.name}\n目录: {project.directory}\n记录图片数: {project.image_count}"
            )

    def _on_run_script(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            self._show_warning("运行脚本", "已有脚本正在运行")
            return

        if self._current_script_path is None:
            self._show_warning("运行脚本", "请先选择脚本")
            return

        if self._is_dirty:
            self._show_warning("运行脚本", "检测到未保存修改，请先保存脚本")
            return

        project = self._get_current_project()
        if project is None:
            self._show_warning("运行脚本", "请先选择数据集")
            return

        dirs = (self._project_manager.get_directories(project.id)
                if self._project_manager else [])
        if not dirs:
            self._show_error("运行失败",
                             "归档内没有有效目录" if project.is_archive_root
                             else f"数据集目录不存在: {project.directory}")
            return
        dataset_dir = dirs[0]

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.setProgram(sys.executable)
        args = [str(self._current_script_path), "--dataset_dir", dataset_dir]
        if len(dirs) > 1:
            args.extend(["--extra_dirs"] + dirs[1:])
        process.setArguments(args)
        process.setWorkingDirectory(str(self._current_script_path.parent))

        process.readyReadStandardOutput.connect(self._on_process_stdout)
        process.readyReadStandardError.connect(self._on_process_stderr)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_process_finished)

        self._process = process
        self._set_running_state(True)

        self._append_log("=" * 50)
        self._append_log(f"开始运行: {self._current_script_path.name}")
        self._append_log(f"Python: {sys.executable}")
        self._append_log(f"数据集目录: {dataset_dir}")
        self._append_log(f"命令参数: --dataset_dir {dataset_dir}")

        process.start()
        started = process.waitForStarted(2000)
        if not started:
            self._set_running_state(False)
            self._show_error("启动失败", "脚本进程启动失败，请检查 Python 环境")
            self._append_log("脚本进程启动失败", level="error")
            self._process = None
            return

        InfoBar.success(
            title="已启动",
            content="脚本正在后台运行",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2000,
        )

    def _on_stop_script(self) -> None:
        if self._process is None or self._process.state() == QProcess.NotRunning:
            return
        self._append_log("收到停止请求，正在终止进程...")
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._append_log("进程未在超时内退出，执行强制结束", level="warning")
            self._process.kill()

    def _on_process_stdout(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_stream_text(text, level="info")

    def _on_process_stderr(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        self._append_stream_text(text, level="error")

    def _append_stream_text(self, text: str, level: str = "info") -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for line in text.split("\n"):
            if line.strip():
                self._append_log(line, level=level)

    def _on_process_error(self, error) -> None:
        self._append_log(f"进程错误: {error}", level="error")

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        self._set_running_state(False)

        if exit_status == QProcess.NormalExit and exit_code == 0:
            self._append_log("脚本运行完成，退出码 0")
            InfoBar.success(
                title="运行完成",
                content="脚本执行成功",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )
        else:
            self._append_log(
                f"脚本运行失败，退出码: {exit_code}，状态: {int(exit_status)}",
                level="error",
            )
            InfoBar.error(
                title="运行失败",
                content=f"脚本退出码: {exit_code}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=5000,
            )

        self._append_log("=" * 50)
        self._process = None

    def _set_running_state(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.save_script_btn.setEnabled(not running)
        self.new_script_btn.setEnabled(not running)
        self.delete_script_btn.setEnabled(not running)

    def _get_current_project(self):
        if self._project_manager is None or self._current_project_id is None:
            return None
        return self._project_manager.get_project(self._current_project_id)

    def _append_log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_map = {
            "info": "INFO",
            "warning": "WARN",
            "error": "ERROR",
        }
        line = f"[{timestamp}] [{level_map.get(level, 'INFO')}] {message}"
        self.log_edit.append(line)
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_edit.setTextCursor(cursor)

    def _clear_log(self) -> None:
        self.log_edit.clear()

    def _show_warning(self, title: str, content: str) -> None:
        InfoBar.warning(
            title=title,
            content=content,
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=3500,
        )

    def _show_error(self, title: str, content: str) -> None:
        InfoBar.error(
            title=title,
            content=content,
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=5000,
        )
        self._append_log(f"{title}: {content}", level="error")
