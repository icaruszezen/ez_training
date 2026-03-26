from PyQt5.QtGui import QImage

import os.path
from enum import Enum

from ez_training.labeling.create_ml_io import CreateMLWriter
from ez_training.labeling.pascal_voc_io import PascalVocWriter
from ez_training.labeling.pascal_voc_io import XML_EXT
from ez_training.labeling.yolo_io import YOLOWriter


class LabelFileFormat(Enum):
    PASCAL_VOC = 1
    YOLO = 2
    CREATE_ML = 3


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.image_path = None
        self.image_data = None
        self.verified = False
        self.lineColor = None
        self.fillColor = None

    def save_create_ml_format(self, filename, shapes, image_path, image_data, class_list, line_color=None, fill_color=None, database_src=None):
        img_folder_name = os.path.basename(os.path.dirname(image_path))
        img_file_name = os.path.basename(image_path)

        image = QImage()
        image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = CreateMLWriter(img_folder_name, img_file_name,
                                image_shape, shapes, filename, local_img_path=image_path)
        writer.verified = self.verified
        writer.write()

    def save_pascal_voc_format(self, filename, shapes, image_path, image_data,
                               line_color=None, fill_color=None, database_src=None):
        img_folder_path = os.path.dirname(image_path)
        img_folder_name = os.path.split(img_folder_path)[-1]
        img_file_name = os.path.basename(image_path)
        if isinstance(image_data, QImage):
            image = image_data
        else:
            image = QImage()
            image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(img_folder_name, img_file_name,
                                 image_shape, local_img_path=image_path)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            difficult = int(shape['difficult'])
            bnd_box = LabelFile.convert_points_to_bnd_box(points)
            writer.add_bnd_box(bnd_box[0], bnd_box[1], bnd_box[2], bnd_box[3], label, difficult)

        writer.save(target_file=filename)

    def save_yolo_format(self, filename, shapes, image_path, image_data, class_list,
                         line_color=None, fill_color=None, database_src=None):
        img_folder_path = os.path.dirname(image_path)
        img_folder_name = os.path.split(img_folder_path)[-1]
        img_file_name = os.path.basename(image_path)
        if isinstance(image_data, QImage):
            image = image_data
        else:
            image = QImage()
            image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = YOLOWriter(img_folder_name, img_file_name,
                            image_shape, local_img_path=image_path)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            difficult = int(shape['difficult'])
            bnd_box = LabelFile.convert_points_to_bnd_box(points)
            writer.add_bnd_box(bnd_box[0], bnd_box[1], bnd_box[2], bnd_box[3], label, difficult)

        writer.save(target_file=filename, class_list=class_list)

    def toggle_verify(self):
        self.verified = not self.verified

    @staticmethod
    def is_label_file(filename):
        file_suffix = os.path.splitext(filename)[1].lower()
        return file_suffix == LabelFile.suffix

    @staticmethod
    def convert_points_to_bnd_box(points):
        if not points:
            return 0, 0, 0, 0
        x_min = float('inf')
        y_min = float('inf')
        x_max = float('-inf')
        y_max = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            x_min = min(x, x_min)
            y_min = min(y, y_min)
            x_max = max(x, x_max)
            y_max = max(y, y_max)

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        return int(x_min), int(y_min), int(x_max), int(y_max)
