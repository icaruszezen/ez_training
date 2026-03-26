import logging
import os

from ez_training.labeling.constants import DEFAULT_ENCODING

logger = logging.getLogger(__name__)

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING


class YOLOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def bnd_box_to_yolo_line(self, box, class_list=None):
        if class_list is None:
            class_list = []
        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, class_list=None, target_file=None):
        if class_list is None:
            class_list = []

        if target_file is None:
            out_path = self.filename + TXT_EXT
            classes_file = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
        else:
            out_path = target_file
            classes_file = os.path.join(os.path.dirname(os.path.abspath(target_file)), "classes.txt")

        with open(out_path, 'w', encoding=ENCODE_METHOD) as out_file:
            for box in self.box_list:
                class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
                out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

        with open(classes_file, 'w', encoding=ENCODE_METHOD) as out_class_file:
            for c in class_list:
                out_class_file.write(c + '\n')


class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        with open(self.class_list_path, 'r', encoding=ENCODE_METHOD) as classes_file:
            self.classes = classes_file.read().strip('\n').split('\n')

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size
        self.verified = False
        self.parse_yolo_format()

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        idx = int(class_index)
        if 0 <= idx < len(self.classes):
            label = self.classes[idx]
        else:
            label = f"class_{idx}"
            logger.warning(
                "class_index %d out of range (classes.txt has %d entries), using '%s'",
                idx, len(self.classes), label,
            )

        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def parse_yolo_format(self):
        with open(self.file_path, 'r', encoding=ENCODE_METHOD) as bnd_box_file:
            for line_no, raw_line in enumerate(bnd_box_file, 1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    logger.warning(
                        "%s:%d: expected 5 fields, got %d – skipping",
                        self.file_path, line_no, len(parts),
                    )
                    continue
                try:
                    class_index, x_center, y_center, w, h = parts[:5]
                    label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(
                        class_index, x_center, y_center, w, h,
                    )
                    self.add_shape(label, x_min, y_min, x_max, y_max, False)
                except (ValueError, IndexError) as e:
                    logger.warning(
                        "%s:%d: failed to parse line – %s",
                        self.file_path, line_no, e,
                    )
