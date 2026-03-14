"""Unified Pascal VOC XML read/write utilities.

Provides canonical functions so that all modules handle VOC format
consistently:
- Coordinates: ``round(float(...))``
- ``difficult`` field is preserved on read/write
- Standard ``xml.etree.ElementTree`` (no lxml dependency)
- Atomic file saving via tempfile
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VocObject:
    """A single object parsed from a VOC XML annotation."""

    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    difficult: bool = False


def parse_voc_objects(xml_path) -> List[VocObject]:
    """Parse all ``<object>`` elements from a VOC XML file.

    Coordinates are converted with ``round(float(...))``.
    Returns an empty list on any parse error.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        logger.warning("VOC XML parse error: %s", xml_path)
        return []

    objects: List[VocObject] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        try:
            xmin = round(float((bnd.findtext("xmin") or "0").strip()))
            ymin = round(float((bnd.findtext("ymin") or "0").strip()))
            xmax = round(float((bnd.findtext("xmax") or "0").strip()))
            ymax = round(float((bnd.findtext("ymax") or "0").strip()))
        except ValueError:
            continue

        difficult = False
        diff_elem = obj.find("difficult")
        if diff_elem is not None and diff_elem.text:
            try:
                difficult = bool(int(diff_elem.text))
            except ValueError:
                pass

        objects.append(VocObject(name, xmin, ymin, xmax, ymax, difficult))
    return objects


def parse_voc_size(xml_path) -> Optional[Tuple[int, int]]:
    """Read ``(width, height)`` from the ``<size>`` element, or *None*."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    size_elem = root.find("size")
    if size_elem is None:
        return None
    try:
        w = int((size_elem.findtext("width") or "0").strip())
        h = int((size_elem.findtext("height") or "0").strip())
    except (ValueError, TypeError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def create_voc_xml(
    folder: str,
    filename: str,
    path: str,
    width: int,
    height: int,
    depth: int = 3,
) -> ET.Element:
    """Build a VOC ``<annotation>`` root element with image metadata."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = path

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size_elem = ET.SubElement(root, "size")
    ET.SubElement(size_elem, "width").text = str(width)
    ET.SubElement(size_elem, "height").text = str(height)
    ET.SubElement(size_elem, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = "0"
    return root


def append_voc_object(
    root: ET.Element,
    label: str,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    difficult: bool = False,
    img_width: int = 0,
    img_height: int = 0,
) -> None:
    """Append an ``<object>`` element to a VOC XML root.

    If *img_width*/*img_height* are provided the ``<truncated>`` flag is
    inferred from whether the box touches an image border.
    """
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = label
    ET.SubElement(obj, "pose").text = "Unspecified"

    truncated = "0"
    if img_width > 0 and img_height > 0:
        if xmin <= 0 or ymin <= 0 or xmax >= img_width or ymax >= img_height:
            truncated = "1"
    ET.SubElement(obj, "truncated").text = truncated
    ET.SubElement(obj, "difficult").text = str(int(difficult))

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)


def save_voc_xml(root: ET.Element, xml_path) -> None:
    """Atomically write a VOC XML tree to *xml_path*."""
    xml_path = Path(xml_path)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    xml_dir = str(xml_path.parent)
    fd, tmp = tempfile.mkstemp(dir=xml_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        Path(tmp).replace(xml_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
