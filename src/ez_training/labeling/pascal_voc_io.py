import codecs
import logging
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

from lxml import etree

from ez_training.labeling.constants import DEFAULT_ENCODING
from ez_training.labeling.ustr import ustr

logger = logging.getLogger(__name__)

XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING


class PascalVocWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace(
            "  ".encode(), "\t".encode())

    def gen_xml(self):
        """Return XML root"""
        if self.filename is None or \
                self.folder_name is None or \
                self.img_size is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.folder_name

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.local_img_path is not None:
            local_img_path = SubElement(top, 'path')
            local_img_path.text = self.local_img_path

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.database_src

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.img_size[1])
        height.text = str(self.img_size[0])
        if len(self.img_size) == 3:
            depth.text = str(self.img_size[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def append_objects(self, top):
        for each_object in self.box_list:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            xmin = int(float(each_object['xmin']))
            ymin = int(float(each_object['ymin']))
            xmax = int(float(each_object['xmax']))
            ymax = int(float(each_object['ymax']))
            img_h = int(float(self.img_size[0]))
            img_w = int(float(self.img_size[1]))
            if xmin <= 0 or ymin <= 0 or xmax >= img_w or ymax >= img_h:
                truncated.text = "1"
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(int(bool(each_object['difficult'])))
            bnd_box = SubElement(object_item, 'bndbox')
            x_min = SubElement(bnd_box, 'xmin')
            x_min.text = str(each_object['xmin'])
            y_min = SubElement(bnd_box, 'ymin')
            y_min.text = str(each_object['ymin'])
            x_max = SubElement(bnd_box, 'xmax')
            x_max.text = str(each_object['xmax'])
            y_max = SubElement(bnd_box, 'ymax')
            y_max.text = str(each_object['ymax'])

    def save(self, target_file=None):
        root = self.gen_xml()
        self.append_objects(root)
        out_path = target_file if target_file is not None else self.filename + XML_EXT
        prettify_result = self.prettify(root)
        with codecs.open(out_path, 'w', encoding=ENCODE_METHOD) as out_file:
            out_file.write(prettify_result.decode('utf8'))


class PascalVocReader:

    def __init__(self, file_path):
        self.shapes = []
        self.file_path = file_path
        self.verified = False
        try:
            self.parse_xml()
        except Exception:
            logger.error("Failed to parse XML: %s", file_path, exc_info=True)

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, bnd_box, difficult):
        x_min = round(float(bnd_box.find('xmin').text))
        y_min = round(float(bnd_box.find('ymin').text))
        x_max = round(float(bnd_box.find('xmax').text))
        y_max = round(float(bnd_box.find('ymax').text))
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def parse_xml(self):
        assert self.file_path.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xml_tree = ElementTree.parse(self.file_path, parser=parser).getroot()
        filename = xml_tree.find('filename').text
        try:
            verified = xml_tree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xml_tree.findall('object'):
            bnd_box = object_iter.find("bndbox")
            label = object_iter.find('name').text
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.add_shape(label, bnd_box, difficult)
        return True
