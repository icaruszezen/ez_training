"""ReferenceImagePanel 单元测试 - Task 2.2: 图片添加功能"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# Ensure a QApplication exists for widget tests
app = QApplication.instance() or QApplication(sys.argv or ["test"])

from ez_training.pages.prelabeling_page import ReferenceImagePanel


class TestValidateImage:
    """_validate_image 方法测试"""

    def setup_method(self):
        self.panel = ReferenceImagePanel()

    def test_valid_jpg(self):
        assert self.panel._validate_image("/tmp/photo.jpg") is True

    def test_valid_jpeg(self):
        assert self.panel._validate_image("/tmp/photo.jpeg") is True

    def test_valid_png(self):
        assert self.panel._validate_image("/tmp/photo.png") is True

    def test_valid_bmp(self):
        assert self.panel._validate_image("/tmp/photo.bmp") is True

    def test_valid_webp(self):
        assert self.panel._validate_image("/tmp/photo.webp") is True

    def test_valid_uppercase_extension(self):
        assert self.panel._validate_image("/tmp/photo.JPG") is True

    def test_valid_mixed_case(self):
        assert self.panel._validate_image("/tmp/photo.Png") is True

    def test_invalid_gif(self):
        assert self.panel._validate_image("/tmp/photo.gif") is False

    def test_invalid_tiff(self):
        assert self.panel._validate_image("/tmp/photo.tiff") is False

    def test_invalid_txt(self):
        assert self.panel._validate_image("/tmp/notes.txt") is False

    def test_invalid_no_extension(self):
        assert self.panel._validate_image("/tmp/photo") is False

    def test_invalid_pdf(self):
        assert self.panel._validate_image("/tmp/doc.pdf") is False


class TestAddImages:
    """add_images 方法测试"""

    def setup_method(self):
        self.panel = ReferenceImagePanel()

    def test_add_single_valid_image(self):
        result = self.panel.add_images(["/tmp/a.jpg"])
        assert result == ["/tmp/a.jpg"]
        assert self.panel.get_image_count() == 1
        assert "/tmp/a.jpg" in self.panel.get_image_paths()

    def test_add_multiple_valid_images(self):
        paths = ["/tmp/a.jpg", "/tmp/b.png", "/tmp/c.webp"]
        result = self.panel.add_images(paths)
        assert result == paths
        assert self.panel.get_image_count() == 3

    def test_skip_invalid_format(self):
        paths = ["/tmp/a.jpg", "/tmp/b.txt", "/tmp/c.png"]
        result = self.panel.add_images(paths)
        assert result == ["/tmp/a.jpg", "/tmp/c.png"]
        assert self.panel.get_image_count() == 2

    def test_skip_duplicate_images(self):
        self.panel.add_images(["/tmp/a.jpg"])
        result = self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        assert result == ["/tmp/b.png"]
        assert self.panel.get_image_count() == 2

    def test_max_limit_enforced(self):
        # Add 10 images (the max)
        paths = [f"/tmp/img{i}.jpg" for i in range(10)]
        result = self.panel.add_images(paths)
        assert len(result) == 10
        assert self.panel.get_image_count() == 10

        # Try to add one more
        result = self.panel.add_images(["/tmp/extra.jpg"])
        assert result == []
        assert self.panel.get_image_count() == 10

    def test_max_limit_partial_add(self):
        # Add 8 images first
        paths = [f"/tmp/img{i}.jpg" for i in range(8)]
        self.panel.add_images(paths)

        # Try to add 5 more - only 2 should succeed
        more = [f"/tmp/extra{i}.png" for i in range(5)]
        result = self.panel.add_images(more)
        assert len(result) == 2
        assert self.panel.get_image_count() == 10

    def test_empty_list(self):
        result = self.panel.add_images([])
        assert result == []
        assert self.panel.get_image_count() == 0

    def test_all_invalid(self):
        result = self.panel.add_images(["/tmp/a.txt", "/tmp/b.pdf"])
        assert result == []
        assert self.panel.get_image_count() == 0

    def test_images_changed_signal_emitted(self):
        received = []
        self.panel.images_changed.connect(lambda paths: received.append(paths))

        self.panel.add_images(["/tmp/a.jpg"])
        assert len(received) == 1
        assert received[0] == ["/tmp/a.jpg"]

    def test_images_changed_signal_not_emitted_when_nothing_added(self):
        received = []
        self.panel.images_changed.connect(lambda paths: received.append(paths))

        self.panel.add_images(["/tmp/a.txt"])  # invalid format
        assert len(received) == 0

    def test_count_label_updated(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        assert "2/10" in self.panel._count_label.text()


class TestRemoveImage:
    """remove_image 方法测试"""

    def setup_method(self):
        self.panel = ReferenceImagePanel()

    def test_remove_existing_image(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png", "/tmp/c.webp"])
        self.panel.remove_image("/tmp/b.png")
        assert self.panel.get_image_count() == 2
        assert "/tmp/b.png" not in self.panel.get_image_paths()
        assert "/tmp/a.jpg" in self.panel.get_image_paths()
        assert "/tmp/c.webp" in self.panel.get_image_paths()

    def test_remove_nonexistent_image_is_noop(self):
        self.panel.add_images(["/tmp/a.jpg"])
        self.panel.remove_image("/tmp/nonexistent.jpg")
        assert self.panel.get_image_count() == 1
        assert "/tmp/a.jpg" in self.panel.get_image_paths()

    def test_remove_from_empty_panel_is_noop(self):
        self.panel.remove_image("/tmp/a.jpg")
        assert self.panel.get_image_count() == 0

    def test_remove_updates_list_widget(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        self.panel.remove_image("/tmp/a.jpg")
        assert self.panel._list_widget.count() == 1
        remaining_item = self.panel._list_widget.item(0)
        assert remaining_item.data(Qt.UserRole) == "/tmp/b.png"

    def test_remove_updates_count_label(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        self.panel.remove_image("/tmp/a.jpg")
        assert "1/10" in self.panel._count_label.text()

    def test_remove_emits_images_changed_signal(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        received = []
        self.panel.images_changed.connect(lambda paths: received.append(paths))
        self.panel.remove_image("/tmp/a.jpg")
        assert len(received) == 1
        assert received[0] == ["/tmp/b.png"]

    def test_remove_nonexistent_does_not_emit_signal(self):
        self.panel.add_images(["/tmp/a.jpg"])
        received = []
        self.panel.images_changed.connect(lambda paths: received.append(paths))
        self.panel.remove_image("/tmp/nonexistent.jpg")
        assert len(received) == 0

    def test_remove_all_images_one_by_one(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        self.panel.remove_image("/tmp/a.jpg")
        self.panel.remove_image("/tmp/b.png")
        assert self.panel.get_image_count() == 0
        assert self.panel.get_image_paths() == []
        assert self.panel._list_widget.count() == 0


class TestClearAll:
    """clear_all 方法测试"""

    def setup_method(self):
        self.panel = ReferenceImagePanel()

    def test_clear_all_removes_all_images(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png", "/tmp/c.webp"])
        self.panel.clear_all()
        assert self.panel.get_image_count() == 0
        assert self.panel.get_image_paths() == []

    def test_clear_all_clears_list_widget(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        self.panel.clear_all()
        assert self.panel._list_widget.count() == 0

    def test_clear_all_updates_count_label(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        self.panel.clear_all()
        assert "0/10" in self.panel._count_label.text()

    def test_clear_all_emits_images_changed_signal(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        received = []
        self.panel.images_changed.connect(lambda paths: received.append(paths))
        self.panel.clear_all()
        assert len(received) == 1
        assert received[0] == []

    def test_clear_all_on_empty_panel(self):
        received = []
        self.panel.images_changed.connect(lambda paths: received.append(paths))
        self.panel.clear_all()
        assert self.panel.get_image_count() == 0
        assert self.panel.get_image_paths() == []
        # Signal should still be emitted even on empty panel
        assert len(received) == 1
        assert received[0] == []

    def test_clear_all_then_add_again(self):
        self.panel.add_images(["/tmp/a.jpg", "/tmp/b.png"])
        self.panel.clear_all()
        result = self.panel.add_images(["/tmp/c.webp"])
        assert result == ["/tmp/c.webp"]
        assert self.panel.get_image_count() == 1
        assert self.panel.get_image_paths() == ["/tmp/c.webp"]
