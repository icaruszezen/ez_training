"""VOCAnnotationWriter 单元测试"""

import os
from pathlib import Path
from xml.etree import ElementTree

import pytest
from PIL import Image

from ez_training.prelabeling.models import BoundingBox
from ez_training.prelabeling.voc_writer import VOCAnnotationWriter


@pytest.fixture
def writer():
    return VOCAnnotationWriter()


@pytest.fixture
def sample_boxes():
    return [
        BoundingBox(label="cat", x_min=10, y_min=20, x_max=100, y_max=200),
        BoundingBox(label="dog", x_min=50, y_min=60, x_max=150, y_max=250, confidence=0.9),
    ]


@pytest.fixture
def sample_image(tmp_path):
    """创建一个临时测试图片"""
    img = Image.new("RGB", (640, 480), color="red")
    img_path = tmp_path / "images" / "test_image.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(img_path))
    return str(img_path)


class TestSaveAnnotation:
    """save_annotation 方法测试"""

    def test_creates_xml_file(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        assert os.path.exists(result)
        assert result.endswith(".xml")

    def test_default_output_path(self, writer, sample_image, sample_boxes):
        """默认输出路径与图片同目录同名 .xml"""
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        expected = str(Path(sample_image).with_suffix(".xml"))
        assert result == expected

    def test_custom_output_path(self, writer, sample_image, sample_boxes, tmp_path):
        custom_path = str(tmp_path / "custom_output.xml")
        result = writer.save_annotation(
            sample_image, (480, 640, 3), sample_boxes, output_path=custom_path
        )
        assert result == custom_path
        assert os.path.exists(custom_path)

    def test_xml_contains_filename(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        tree = ElementTree.parse(result)
        root = tree.getroot()
        assert root.find("filename").text == "test_image.jpg"

    def test_xml_contains_folder(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        tree = ElementTree.parse(result)
        root = tree.getroot()
        assert root.find("folder").text == "images"

    def test_xml_contains_image_size(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        tree = ElementTree.parse(result)
        root = tree.getroot()
        size = root.find("size")
        assert size.find("width").text == "640"
        assert size.find("height").text == "480"
        assert size.find("depth").text == "3"

    def test_xml_contains_all_boxes(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        tree = ElementTree.parse(result)
        root = tree.getroot()
        objects = root.findall("object")
        assert len(objects) == 2

    def test_xml_box_labels(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        tree = ElementTree.parse(result)
        root = tree.getroot()
        objects = root.findall("object")
        labels = [obj.find("name").text for obj in objects]
        assert labels == ["cat", "dog"]

    def test_xml_box_coordinates(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        tree = ElementTree.parse(result)
        root = tree.getroot()
        obj = root.findall("object")[0]
        bndbox = obj.find("bndbox")
        assert bndbox.find("xmin").text == "10"
        assert bndbox.find("ymin").text == "20"
        assert bndbox.find("xmax").text == "100"
        assert bndbox.find("ymax").text == "200"

    def test_empty_boxes(self, writer, sample_image):
        """空边界框列表应生成无 object 的 XML"""
        result = writer.save_annotation(sample_image, (480, 640, 3), [])
        tree = ElementTree.parse(result)
        root = tree.getroot()
        assert len(root.findall("object")) == 0
        assert root.find("filename").text == "test_image.jpg"

    def test_returns_path_string(self, writer, sample_image, sample_boxes):
        result = writer.save_annotation(sample_image, (480, 640, 3), sample_boxes)
        assert isinstance(result, str)


class TestGetImageSize:
    """get_image_size 方法测试"""

    def test_rgb_image(self, writer, tmp_path):
        img = Image.new("RGB", (800, 600))
        path = str(tmp_path / "rgb.png")
        img.save(path)
        h, w, d = writer.get_image_size(path)
        assert (h, w, d) == (600, 800, 3)

    def test_grayscale_image(self, writer, tmp_path):
        img = Image.new("L", (320, 240))
        path = str(tmp_path / "gray.png")
        img.save(path)
        h, w, d = writer.get_image_size(path)
        assert (h, w, d) == (240, 320, 1)

    def test_rgba_image(self, writer, tmp_path):
        img = Image.new("RGBA", (1024, 768))
        path = str(tmp_path / "rgba.png")
        img.save(path)
        h, w, d = writer.get_image_size(path)
        assert (h, w, d) == (768, 1024, 4)

    def test_jpeg_format(self, writer, tmp_path):
        img = Image.new("RGB", (1920, 1080))
        path = str(tmp_path / "photo.jpg")
        img.save(path)
        h, w, d = writer.get_image_size(path)
        assert (h, w, d) == (1080, 1920, 3)

    def test_bmp_format(self, writer, tmp_path):
        img = Image.new("RGB", (400, 300))
        path = str(tmp_path / "image.bmp")
        img.save(path)
        h, w, d = writer.get_image_size(path)
        assert (h, w, d) == (300, 400, 3)
