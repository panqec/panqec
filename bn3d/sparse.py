from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np


def zero_row(n_cols: int):
    """Create a zero sparse row in the csr format"""

    return csr_matrix(([], ([], [])), shape=(1, n_cols), dtype='uint8')


def empty_row(n_cols: int):
    """Create an empty sparse row in the csr format"""

    return csr_matrix(([], ([], [])), shape=(0, n_cols), dtype='uint8')


def is_empty(matrix):
    return matrix.shape[0] == 0


def insert_mod2(index: int, row_matrix):
    """Insert 1 at 'index' in the row matrix, modulo 2"""

    # Check that matrix is a row matrix
    if len(row_matrix.shape) != 2 or row_matrix.shape[0] != 1:
        raise ValueError(f"The input should be a row matrix, not a {row_matrix.shape}-matrix")

    # Check that the row only contains 1s
    if not np.all(row_matrix.data == 1):
        raise ValueError("The row matrix should only contain 0 and 1")

    # If matrix[index] is not zero, put it to zero
    if index in row_matrix.indices:
        row_matrix.indices = np.setdiff1d(row_matrix.indices, [index])
    else:
        row_matrix.indices = np.append(row_matrix.indices, [index])

    row_matrix.data = np.ones(len(row_matrix.indices), dtype='uint8')
    row_matrix.indptr = [0, len(row_matrix.indices)]


def is_one(index: int, row_matrix):
    """Return True if row_matrix[index] is nonzero"""

    return index in row_matrix.indices


def vstack(matrices):
    """"""
    return sparse.vstack(matrices, format='csr', dtype='uint8')


def hstack(matrices):
    """"""
    return sparse.hstack(matrices, format='csr', dtype='uint8')


def hsplit(matrix):
    indices = np.array(matrix.indices)
    n = len(indices // 2)
    a_indices = indices[indices < n]
    b_indices = indices[indices >= n]

    a_indptr = [0, len(n)]
    b_indptr = [0, len(n)]

    a_data = np.ones(n, dtype='uint8')
    b_data = np.ones(n, dtype='uint8')

    a = csr_matrix((a_data, a_indices, a_indptr), dtype='uint8')
    b = csr_matrix((b_data, b_indices, b_indptr), dtype='uint8')

    return a, b
