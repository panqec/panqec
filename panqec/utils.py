"""
Miscellaneous useful utilities.

Many of these are copied from the internet.
"""
import gzip
import hashlib
from itertools import product
import json
import os
from typing import Dict, Any, Optional, Callable

from ldpc import bposd_decoder
from ldpc.mod2 import rank, nullspace
import networkx as nx
import numpy as np
from pyvis.network import Network
from scipy.sparse import csr_array

from panqec.bsparse import is_sparse, to_array


def log(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def get_all_logicals(Hx, Hz, k=None, reduce_iter=0, osd_order=6, verbose=False):
    arbitrary_x_logicals = get_arbitrary_logicals(Hx, Hz, 'X', k, verbose=verbose)

    min_z_logicals = get_minimum_logicals(
        Hx, arbitrary_x_logicals, osd_order=osd_order, verbose=verbose
    )
    min_x_logicals = get_minimum_logicals(
        Hz, min_z_logicals, osd_order=osd_order, verbose=verbose
    )

    min_x_logicals = np.array(min_x_logicals, dtype='uint8')
    min_z_logicals = np.array(min_z_logicals, dtype='uint8')

    for i in range(min_x_logicals.shape[0]):
        min_x_logicals[i] = reduce_logical(Hx, min_x_logicals[i], reduce_iter, random=False)

    for i in range(min_z_logicals.shape[0]):
        min_z_logicals[i] = reduce_logical(Hz, min_z_logicals[i], reduce_iter, random=False)

    return {'X': min_x_logicals, 'Z': min_z_logicals}


def reduce_logical(H, logical, n_iter, random=False):
    current_weight = len(logical.nonzero()[0])

    for i_iter in range(n_iter):
        for i in range(H.shape[0]):
            if random:
                stab_i = np.random.randint(0, H.shape[0])
            else:
                stab_i = i

            stab = H[stab_i]
            new_logical = np.logical_xor(logical, stab)
            new_weight = len(new_logical.nonzero()[0])

            if new_weight < current_weight:
                logical = new_logical
                current_weight = new_weight
                # print(f"New weight {current_weight} at iter {i_iter}")

            elif current_weight == new_weight:
                if np.random.rand() < 0.5:
                    logical = new_logical

    # print('end')

    return logical


def get_minimum_logicals(H, anti_logical_basis, osd_order=6, verbose=False):
    if len(anti_logical_basis) == 0:
        return []

    H_with_logical = np.vstack([H, anti_logical_basis])

    decoder = bposd_decoder(
        H_with_logical,
        error_rate=0.01,
        max_iter=200,
        bp_method='msl',
        osd_method="osd0",
        osd_order=osd_order
    )

    logicals = []
    for i in range(len(anti_logical_basis)):
        log(f"Finding logicals {i+1}/{len(anti_logical_basis)}  ", end='\r', verbose=verbose)
        logical_syndrome = [0 for _ in range(len(anti_logical_basis))]
        logical_syndrome[i] = 1

        syndrome = np.array([0] * H.shape[0] + logical_syndrome)

        predicted_logical_binary = decoder.decode(syndrome)

        logicals.append(predicted_logical_binary)

    return logicals


def get_arbitrary_logicals(Hx, Hz, pauli='X', k=None, verbose=False):
    if verbose:
        log(f"Get logicals {pauli}", verbose=verbose)

    if pauli == 'X':
        H = [Hx, Hz]
    else:
        H = [Hz, Hx]

    rank_H0 = rank(H[0])
    augmented_H = nullspace(H[1])

    if k is None:
        k = Hx.shape[1] - rank_H0 - rank(H[1])

    logicals = []
    for i, s in enumerate(augmented_H):
        log(f"{i} / {len(augmented_H)}", end="\r", verbose=verbose)

        new_H0 = np.vstack([H[0], s])
        if rank(new_H0) > rank_H0:
            H[0] = new_H0
            rank_H0 += 1

            logicals.append(s)

        if len(logicals) == k:
            break

    log(verbose=verbose)
    log("Done", verbose=verbose)

    return logicals


def is_multiorthogonal(
    H: np.ndarray,
    L: np.ndarray,
    k: int = 3,
    levels: Optional[list[int]] = None,
    verbose: bool = False
):
    if levels is None:
        levels = list(range(k))

    multiorthogonal = True
    multiorthogonal_logicals = dict()
    for n_L in levels:
        indices = list(product(
            *[range(L.shape[0]) for _ in range(n_L)],
            *[range(H.shape[0]) for _ in range(k-n_L)]
        ))

        for logical_indices in product(*[range(L.shape[0]) for _ in range(n_L)]):
            multiorthogonal_logicals[logical_indices] = True

        for i, index in enumerate(indices):
            if i % 2 == 0:  # used to reduce the print rate (useful in notebooks)
                log(
                    f"{k}-orthogonality with {n_L} logicals: "
                    f"{round((i+1) / len(indices) * 100)}%",
                    end='\r',
                    verbose=verbose
                )

            logical_rows = [L[j] for j in index[:n_L]]
            stab_rows = [H[j] for j in index[n_L:]]

            prod = np.prod([*logical_rows, *stab_rows], axis=0)

            expected_value = 0 if n_L < k else 1

            # if np.sum(prod) % 2 == expected_value:
            #     print(f"\nIt works for {index}\n")

            if np.sum(prod) % 2 != expected_value:
                multiorthogonal_logicals[index[:n_L]] = False
                log(f"Not {k}-orthogonal when including {n_L} logicals", verbose=verbose)
                log(f"Logicals {index[:n_L]} and stabilizers {index[n_L:]} anticommute",
                    verbose=verbose)
                multiorthogonal = False
            else:
                log(f"Logicals {index[:n_L]} and stabilizers {index[n_L:]} commute",
                    verbose=verbose)

    logical_triplets = []
    for logical_indices in multiorthogonal_logicals.keys():
        if multiorthogonal_logicals[logical_indices]:
            triplet = tuple(sorted(logical_indices))
            if triplet not in logical_triplets:
                logical_triplets.append(triplet)

    return multiorthogonal


def quadratic(x, *params):
    p_th, nu, A, B, C = params
    return A + B*x + C*x**2


def rescale_prob(x_data, *params):
    """Rescaled physical error rate."""
    p, d = x_data
    p_th, nu, A, B, C = params
    x = (p - p_th)*d**nu
    return x


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


def fmt_confidence_interval(
    estimate, left, right, sn=None, unit=None, sn_cutoff=8,
):
    """Format confidence interval for latex."""
    x = estimate
    dx_right = right - estimate
    dx_left = estimate - left
    n_decimals = min(
        -int(np.floor(np.log10(np.abs(dx_left)))),
        -int(np.floor(np.log10(np.abs(dx_right)))),
    )
    leading_magnitude = np.abs(min(dx_left, dx_right))/10**-n_decimals
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
        dx_mag_left = np.abs(dx_left)/10**exponent
        dx_mag_right = np.abs(dx_right)/10**exponent
    else:
        exponent = 0
        x_mag = np.abs(x)
        dx_mag_left = np.abs(dx_left)
        dx_mag_right = np.abs(dx_right)
    x_round = np.round(x_mag, decimals=n_decimals + exponent)
    dx_left_round = np.round(dx_mag_left, decimals=n_decimals + exponent)
    dx_right_round = np.round(dx_mag_right, decimals=n_decimals + exponent)
    if min(dx_left_round, dx_right_round) > 1.5:
        x_str = str(int(x_round))
    else:
        x_str = str(x_round)
    dx_left_str = str(dx_left_round)
    dx_right_str = str(dx_right_round)

    if sn:
        fmt_str = r'{%s}^{+%s}_{-%s}\times {10}^{%s}' % (
            x_str, dx_right_str, dx_left_str, exponent
        )
        if dx_right_str == dx_left_str:
            fmt_str = r'{(%s\pm %s)}\times {10}^{%s}' % (
                x_str, dx_right_str, exponent
            )
        if x < 0:
            fmt_str = f'-{fmt_str}'
    else:
        fmt_str = r'{%s}^{+%s}_{-%s}' % (x_str, dx_right_str, dx_left_str)
        if dx_left_str == dx_right_str:
            fmt_str = r'{%s}\pm %s' % (x_str, dx_right_str)
    if unit is not None:
        if r'\pm' in fmt_str:
            fmt_str = f'({fmt_str})'
        if unit == r'\%':
            fmt_str += r'\mathrm{%s}' % unit
        else:
            fmt_str += r'\ \mathrm{%s}' % unit
    return f'${fmt_str}$'


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


def dict_where(signs):
    return set([k for k, v in signs.items() if v])


def face_coords(axis_xyz_list, size):
    """Convert list of (axis, x, y, z) into coordinates."""
    faces = []
    for axis_xyz in axis_xyz_list:
        i, x, y, z = axis_xyz
        lim = 2*np.array(size, dtype=int)
        diff = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int)
        new = np.mod(2*np.array([x, y, z], dtype=int) + diff[i], lim)
        faces.append(tuple(new))
    return faces


def edge_coords(axis_xyz_list, size):
    """Convert list of (axis, x, y, z) into coordinates."""
    edges = []
    for axis_xyz in axis_xyz_list:
        i, x, y, z = axis_xyz
        lim = 2*np.array(size, dtype=int)
        diff = np.eye(3, dtype=int)
        new = np.mod(2*np.array([x, y, z], dtype=int) + diff[i], lim)
        edges.append(tuple(new))
    return edges


def format_polynomial(symbol, coefficients, digits=3):
    """Polynomial as nice-looking string."""
    poly_str = ''
    format_str = '{:.%df}' % digits
    for power, coefficient in enumerate(coefficients):
        if coefficient < 0:
            sign = '-'
        else:
            sign = '+'
        magnitude = format_str.format(abs(coefficient))
        if power == 0:
            monomial = ''
        elif power == 1:
            monomial = symbol
        else:
            monomial = f'{symbol}^{power}'
        if coefficient != 0:
            if power == 0:
                if sign == '+':
                    poly_str += f'{magnitude}{monomial}'
                else:
                    poly_str += f'{sign}{magnitude}{monomial}'
            else:
                poly_str += f' {sign} {magnitude}{monomial}'
    return poly_str


def simple_print(a, zeros=True):
    """Print binary array in space-saving manner.

    Examples
    --------
    >>> simple_print(np.eye(3))
    100
    010
    001

    >>> simple_print(np.eye(3), zeros=False)
    1
     1
      1
    """
    if is_sparse(a):
        dense = to_array(a)
    else:
        dense = a

    if len(dense.shape) == 1:
        dense = np.array([dense])

    for row in dense:
        row_str = ''
        for entry in row:
            if entry == 0:
                if zeros:
                    row_str += '0'
                else:
                    row_str += ' '
            else:
                row_str += '1'
        print(row_str.rstrip())


def find_nearest(array, value):
    """Find the nearest element in array to given value.

    Parameters
    ----------
    array : Union[List, np.ndarray]
        The array with values in it.
    value : Union[float, int]
        The value to compare to.

    Returns
    -------
    nearest_value : Union[float, int]
        The element in the array that is nearest to the given value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def progress_bar(k, n, bar_length=20):
    ratio = k / n
    percentage = str(int(ratio * 100)) + "%"
    n_progress = int(ratio * bar_length)
    n_left = int((1-ratio) * bar_length)

    print(f"{percentage}|" + "█" * n_progress + " " * n_left + f"| {k}/{n}")


def load_json(file):
    if not os.path.isfile(file):
        raise ValueError(f"File {file} not found")
    if os.path.splitext(file)[-1] not in ['.json', '.gz']:
        raise ValueError(f"File {file} is not a .json or .json.gz file")

    if os.path.splitext(file)[-1] == '.json':
        with open(file) as f:
            data = json.load(f)
    elif os.path.splitext(file)[-1] == '.gz':
        with gzip.open(file, 'rb') as gz:
            data = json.loads(gz.read().decode('utf-8'))

    return data


def save_json(data, file):
    if os.path.splitext(file)[-1] not in ['.json', '.gz']:
        raise ValueError(f"Specified file {file} should have extension"
                         ".json or .json.gz")

    if isinstance(data, np.ndarray):
        data = data.tolist()

    if os.path.splitext(file)[-1] == '.json':
        with open(file, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)
    else:
        with gzip.open(file, 'wb') as gz:
            gz.write(json.dumps(data, cls=NumpyEncoder).encode('utf-8'))


def get_label(name: str, parameters: Dict[str, Any]) -> str:
    """Get label filtering entries from class name and parameters.

    Parameters
    ----------
    name : str
        The name of the class.
    parameters : Dict[str, Any]
        The dictionary of parameters used to initialize an instance of that
        class.

    Returns
    -------
    label : str
        The formatted label.

    Examples
    --------
    >>> get_label('Toric2DCode', {'L_x': 5, 'L_y': 6, 'L_z': 7})
    'Toric2DCode(L_x=5, L_y=6, L_z=7)'
    """
    label = ''
    parameter_labels = []
    for key, value in parameters.items():
        formatted_value = repr(value)
        if isinstance(value, float):
            formatted_value = str(np.round(value, 6))
        parameter_labels.append(f'{key}={formatted_value}')
    label = name + '(' + ', '.join(parameter_labels) + ')'
    return label


def draw_Tanner_graph(H: np.ndarray, notebook: bool = True):
    """Draw the Tanner graph of a classical code using Pyvis.

    To draw on a notebook, use notebook=True, store the output
    in a variable `nt` and run `nt.show('nx.html')`
    """

    graph = nx.algorithms.bipartite.from_biadjacency_matrix(
        csr_array(H)
    )
    node_type = [graph.nodes[i]['bipartite'] for i in range(len(graph.nodes))]

    graph = nx.convert_node_labels_to_integers(graph)

    nt = Network(notebook=notebook, cdn_resources='in_line')
    nt.from_nx(graph)

    n_checks = len(node_type) - sum(node_type)

    for edge in nt.get_edges():
        edge['width'] = 8

    for node_id in nt.get_nodes():
        node = nt.get_node(node_id)
        if node_type[node_id] == 0:
            node['label'] = str(node['id'])
        else:
            node['label'] = str(node['id'] - n_checks)
        node['shape'] = ['box', 'circle'][node_type[node_id]]
        if not node_type[node_id]:
            node['color'] = 'orange'

        node['font'] = {'size': 45}

    nt.show('nx.html')

    return nt
