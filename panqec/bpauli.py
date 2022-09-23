"""
Bit array or vector representations of Paulis for 3Di codes.

Although qecsim already has such an implementation, some of these extra
routines are useful specifically for dealing with the 3D code.

:Author:
    Eric Huang
"""
from typing import Union, List
import numpy as np
from . import bsparse
from scipy.sparse import csr_matrix


def bs_prod(a, b) -> np.ndarray:
    """Array of 0 for commutes and 1 for anticommutes bvectors."""

    # If lists, convert to numpy
    if isinstance(a, list):
        a = np.array(a, dtype='uint8')
    if isinstance(b, list):
        b = np.array(b, dtype='uint8')

    # Determine the output shape.
    # In particular, flatten array where needed.
    output_shape = None
    if len(a.shape) == 2 and len(b.shape) == 1:
        output_shape = a.shape[0]
    elif len(a.shape) == 1 and len(b.shape) == 2:
        output_shape = b.shape[0]

    # If only singles, then convert to 2D array.
    if len(a.shape) == 1:
        a = np.reshape(a, (1, a.shape[0]))
    if len(b.shape) == 1:
        b = np.reshape(b, (1, b.shape[0]))

    # Check the shapes are correct.
    if a.shape[1] % 2 != 0:
        raise ValueError(
            f'Length {a.shape[1]} binary vector not of even length.'
        )
    if b.shape[1] % 2 != 0:
        raise ValueError(
            f'Length {b.shape[1]} binary vector not of even length.'
        )
    if b.shape[1] != a.shape[1]:
        raise ValueError(
            f'Length {a.shape[1]} bvector cannot be '
            f'composed with length {b.shape[1]}'
        )

    if bsparse.is_sparse(a) or bsparse.is_sparse(b):
        commutes = _bs_prod_sparse(a, b)
    else:
        # Number of qubits.
        n = int(a.shape[1]/2)

        # Commute commutator by binary symplectic form.
        a_X = a[:, :n]
        a_Z = a[:, n:]
        b_X = b[:, :n]
        b_Z = b[:, n:]

        commutes = (a_X.dot(b_Z.T) + a_Z.dot(b_X.T)) % 2

    if output_shape is not None:
        commutes = commutes.reshape(output_shape)

    return commutes


def _bs_prod_sparse(a, b):
    """Array of 0 for commutes and 1 for anticommutes bvectors."""

    # Commute commutator by binary symplectic form.
    n = int(a.shape[1]/2)

    if not bsparse.is_sparse(a):
        a = bsparse.from_array(a)
    if not bsparse.is_sparse(b):
        b = bsparse.from_array(b)

    a_X = a[:, :n]
    a_Z = a[:, n:]
    b_X = b[:, :n]
    b_Z = b[:, n:]

    commutes = (a_X.dot(b_Z.T) + a_Z.dot(b_X.T))
    commutes.data %= 2

    if commutes.shape[0] == 1:
        return commutes.toarray()[0, :]
    elif commutes.shape[1] == 1:
        return commutes.toarray()[:, 0]
    else:
        return commutes.toarray()


def pauli_to_bsf(error_pauli):
    ps = np.array(list(error_pauli))
    xs = (ps == 'X') + (ps == 'Y')
    zs = (ps == 'Z') + (ps == 'Y')

    error = np.hstack((xs, zs)).astype('uint8')

    return error


def pauli_string_to_bvector(pauli_string: str) -> np.ndarray:
    X_block = []
    Z_block = []
    for character in pauli_string:
        if character == 'I':
            X_block.append(0)
            Z_block.append(0)
        elif character == 'X':
            X_block.append(1)
            Z_block.append(0)
        elif character == 'Y':
            X_block.append(1)
            Z_block.append(1)
        elif character == 'Z':
            X_block.append(0)
            Z_block.append(1)
    bvector = np.concatenate([X_block, Z_block]).astype(np.uint)
    return bvector


def bvector_to_pauli_string(bvector: np.ndarray) -> str:
    n = int(bvector.shape[0]/2)
    pauli_string = ''
    for i in range(n):
        pauli_string += {
            (0, 0): 'I',
            (1, 0): 'X',
            (1, 1): 'Y',
            (0, 1): 'Z'
        }[(bvector[i], bvector[i + n])]
    return pauli_string


def get_effective_error(
    total_error,
    logicals_x,
    logicals_z,
) -> np.ndarray:
    """Effective Pauli error on logical qubits after decoding."""

    if logicals_x.shape != logicals_z.shape:
        raise ValueError('Logical Xs and Zs must be of same shape.')

    # Number of pairs of logical operators.
    if len(logicals_x.shape) == 1:
        n_logical = 1
    else:
        n_logical = int(logicals_x.shape[0])

    # Get the number of total errors given.
    num_total_errors = 1
    if len(total_error.shape) > 1:
        num_total_errors = total_error.shape[0]

    # The shape of the array to be returned.
    final_shape: tuple = (num_total_errors, 2*n_logical)
    if num_total_errors == 1:
        final_shape = (2*n_logical, )

    effective_Z = bs_prod(logicals_x, total_error)
    effective_X = bs_prod(logicals_z, total_error)

    if num_total_errors == 1:
        effective = np.concatenate([effective_X, effective_Z])
    elif n_logical == 1:
        effective = np.array([effective_X, effective_Z]).T
    else:
        effective = np.array([
            np.concatenate([effective_X[:, i], effective_Z[:, i]])
            for i in range(num_total_errors)
        ])

    # Flatten the array if only one total error is given.
    effective = effective.reshape(final_shape)
    return effective


def bvector_to_int(bvector: np.ndarray) -> int:
    """Convert bvector to integer for effecient storage."""
    return int(''.join(map(str, bvector)), 2)


def int_to_bvector(int_rep: int, n: int) -> np.ndarray:
    """Convert integer representation to n-qubit Pauli bvector."""
    binary_string = ('{:0%db}' % (2*n)).format(int_rep)
    bvector = np.array(tuple(binary_string), dtype=np.uint)
    return bvector


def bvectors_to_ints(bvector_list: list) -> list:
    """List of bvectors to integers for efficient storage."""
    return list(map(
        bvector_to_int,
        bvector_list
    ))


def ints_to_bvectors(int_list: list, n: int) -> list:
    """Convert list of integers back to bvectors."""
    bvectors = []
    for int_rep in int_list:
        bvectors.append(int_to_bvector(int_rep, n))
    return bvectors


def gf2_rank(rows):
    """Find rank of a matrix over GF2 given as list of binary ints.

    From https://stackoverflow.com/questions/56856378
    """
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank


def brank(matrix):
    """Rank of a binary matrix."""

    matrix = bsparse.to_array(matrix)

    # Convert to list of binary numbers.
    rows = [int(''.join(map(str, row)), 2) for row in matrix.astype(int)]
    return gf2_rank(rows)


def apply_deformation(
    deformation_indices: Union[List[bool], np.ndarray], bsf: np.ndarray
) -> np.ndarray:
    """Return Hadamard-deformed bsf at given indices."""
    n = len(deformation_indices)
    deformed = np.zeros_like(bsf)
    if len(bsf.shape) == 1:
        if bsf.shape[0] != 2*n:
            raise ValueError(
                f'Deformation index length {n} does not match '
                f'bsf shape {bsf.shape}, which should be {(2*n,)}'
            )
        for i, deform in enumerate(deformation_indices):
            if deform:
                deformed[i] = bsf[i + n]
                deformed[i + n] = bsf[i]
            else:
                deformed[i] = bsf[i]
                deformed[i + n] = bsf[i + n]
    else:
        if bsf.shape[1] != 2*n:
            raise ValueError(
                f'Deformation index length {n} does not match '
                f'bsf shape {bsf.shape}, which should be '
                f'{(bsf.shape[0], 2*n)}.'
            )
        for i, deform in enumerate(deformation_indices):
            if deform:
                deformed[:, i] = bsf[:, i + n]
                deformed[:, i + n] = bsf[:, i]
            else:
                deformed[:, i] = bsf[:, i]
                deformed[:, i + n] = bsf[:, i + n]
    return deformed


def bsf_wt(bsf):
    """
    Return weight of given binary symplectic form.
    :param bsf: Binary symplectic vector or matrix.
    :type bsf: numpy.array (1d or 2d) or csr_matrix
    :return: Weight
    :rtype: int
    """
    if isinstance(bsf, np.ndarray):
        assert np.array_equal(bsf % 2, bsf), \
                'BSF {} is not in binary form'.format(bsf)
        return np.count_nonzero(sum(np.hsplit(bsf, 2)))

    elif isinstance(bsf, csr_matrix):
        assert np.all(bsf.data == 1), \
                'BSF {} is not in binary form'.format(bsf)

        n = bsf.shape[1] // 2
        x_indices = bsf.indices[bsf.indices < n]
        z_indices = bsf.indices[bsf.indices >= n] - n

        return len(np.union1d(x_indices, z_indices))
    else:
        raise TypeError(
            f"bsf matrix should be a numpy array or "
            f"csr_matrix, not {type(bsf)}"
        )


def bsf_to_pauli(bsf):
    """
    Convert the given binary symplectic form to Pauli operator(s).
    (1 0 0 0 1 | 0 0 1 0 1) -> XIZIY
    Assumptions:
    * bsf is a numpy.array (1d or 2d) in binary symplectic form.
    :param bsf: Binary symplectic vector or matrix.
    :type bsf: numpy.array (1d or 2d)
    :return: Pauli operators.
    :rtype: str or list of str
    """

    if isinstance(bsf, np.ndarray):
        assert np.array_equal(bsf % 2, bsf), \
                'BSF {} is not in binary form'.format(bsf)

        def _to_pauli(b, t=str.maketrans('0123', 'IXZY')):  # noqa: B008,E501 (deliberately reuse t)
            xs, zs = np.hsplit(b, 2)
            ps = (xs + zs * 2).astype(str)  # 0=I, 1=X, 2=Z, 3=Y
            return ''.join(ps).translate(t)

        if bsf.ndim == 1:
            return _to_pauli(bsf)
        else:
            return [_to_pauli(b) for b in bsf]
    else:
        assert np.all(bsf.data == 1), \
                'BSF {} is not in binary form'.format(bsf)

        def _to_pauli(b):  # type:ignore
            n = bsf.shape[1] // 2
            pauli_string = ['I' for _ in range(n)]
            for i in b.indices:
                if i < n:
                    pauli_string[i] = 'X'
                elif i >= n:
                    if pauli_string[i - n] == 'X':
                        pauli_string[i - n] = 'Y'
                    else:
                        pauli_string[i - n] = 'Z'

            return ''.join(pauli_string)

        return [_to_pauli(b) for b in bsf]
