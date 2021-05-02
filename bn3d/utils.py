"""
Micellaneous useful utilities.

Many of these are copied from the internet.

:Author:
    Eric Huang
"""
import numpy as np
import json
from typing import Callable


def sizeof_fmt(num, suffix='B'):
    """Size to human readable format.

    From stack overflow.
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def nested_map(function: Callable):
    """Return function that applies function to nested lists."""
    def mapper(item):
        if isinstance(item, list):
            return [mapper(x) for x in item]
        else:
            return function(item)

    return mapper


def identity(x):
    return x


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
