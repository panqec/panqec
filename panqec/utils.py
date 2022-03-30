"""
Micellaneous useful utilities.

Many of these are copied from the internet.

:Author:
    Eric Huang
"""
import numpy as np
import json
import hashlib
from typing import Callable


def get_direction_from_bias_ratio(pauli: str, eta) -> dict:
    """Get noise params given Pauli and bias."""

    if eta == np.inf:
        r_bias = 1.
    else:
        r_bias = eta / (1 + eta)
    r_other = (1 - r_bias) / 2

    params: dict = {}

    if pauli == 'Z':
        params = {'r_x': r_other, 'r_y': r_other, 'r_z': r_bias}
    elif pauli == 'X':
        params = {'r_x': r_bias, 'r_y': r_other, 'r_z': r_other}
    elif pauli == 'Y':
        params = {'r_x': r_other, 'r_y': r_bias, 'r_z': r_other}

    return params


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


def list_where_str(array):
    return ' '.join(map(
        lambda x: ''.join(map(str, x)),
        sorted(map(tuple, np.array(np.where(array)).T))
    ))


def list_where(array):
    """Get locations of binary list as sorted list of tuples."""
    return sorted(map(tuple, np.array(np.where(array)).T))


def set_where(array):
    """Get locations of binary list as sorted list of tuples."""
    return set(map(tuple, np.array(np.where(array)).T))


def fmt_uncertainty(x, dx, sn=None, sn_cutoff=8, unit=None):
    """Format uncertainty for latex."""
    n_decimals = -int(np.floor(np.log10(np.abs(dx))))
    leading_magnitude = np.abs(dx)/10**-n_decimals
    if leading_magnitude <= 1.5:
        n_decimals += 1
    if sn is None:
        if np.abs(x) >= 10**sn_cutoff or np.abs(x) <= 10**-sn_cutoff:
            sn = True
        else:
            sn = False
    if sn:
        exponent = int(np.floor(np.log10(np.abs(x))))
        x_mag = np.abs(x)/10**exponent
        dx_mag = np.abs(dx)/10**exponent
    else:
        exponent = 0
        x_mag = np.abs(x)
        dx_mag = np.abs(dx)
    x_round = np.round(x_mag, decimals=n_decimals + exponent)
    dx_round = np.round(dx_mag, decimals=n_decimals + exponent)
    if dx_round > 1.5:
        x_str = str(int(x_round))
    else:
        x_str = str(x_round)
    dx_str = str(dx_round)

    if sn:
        fmt_str = r'(%s \pm %s)\times {10}^{%s}' % (
            x_str, dx_str, exponent
        )
        if x < 0:
            fmt_str = f'-{fmt_str}'
    else:
        fmt_str = r'%s \pm %s' % (x_str, dx_str)
        if x < 0:
            fmt_str = f'-({fmt_str})'
    if unit is not None:
        if '(' not in fmt_str:
            fmt_str = f'({fmt_str})'
        fmt_str += r'\ \mathrm{%s}' % unit
    fmt_str = f'${fmt_str}$'
    return fmt_str


def hash_json(dictionary):
    """Produce MD5 hash of dictionary"""
    dict_no_hash = {
        k: v for k, v in dictionary.items()
        if k != 'hash'
    }
    json_string = json.dumps(dict_no_hash, sort_keys=True, indent=2)
    return hashlib.md5(json_string.encode('utf-8')).hexdigest()
