import sys
from ez_training.labeling.constants import DEFAULT_ENCODING


def ustr(x):
    """py2/py3 unicode helper"""
    if sys.version_info < (3, 0, 0):
        if type(x) == str:
            return x.decode(DEFAULT_ENCODING)
        return x
    else:
        return x
