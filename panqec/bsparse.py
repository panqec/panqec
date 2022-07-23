"""
Module to deal with binary sparse matrices, in the compressed sparse row (CSR)
format.

It is mostly a wrapper around scipy.sparse, making sure that everything is in
the correct format and is done as efficiently as possible.
"""

from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np


def zero_row(n_cols: int):
    """Create a zero sparse row in the csr format"""

    return csr_matrix((np.array([]), (np.array([]), np.array([]))),
                      shape=(1, n_cols), dtype='uint8')


def zero_matrix(shape):
    """Create a zero sparse matrix in the csr format"""

    return csr_matrix((np.array([]), (np.array([]), np.array([]))),
                      shape=shape, dtype='uint8')


def empty_row(n_cols: int):
    """Create an empty sparse row in the csr format"""

    return csr_matrix(([], ([], [])), shape=(0, n_cols), dtype='uint8')


def from_array(array):
    if isinstance(array, list):
        array = np.array(array)
    sparse_matrix = csr_matrix(array, dtype='uint8')
    return sparse_matrix


def to_array(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    else:
        return matrix.toarray()


def is_empty(matrix):
    return matrix.shape[0] == 0


def is_sparse(matrix):
    return type(matrix) == csr_matrix


def is_one(index: int, row_matrix):
    """Return True if row_matrix[index] is nonzero"""

    return index in row_matrix.indices


def insert_mod2(index: int, row_matrix):
    """Insert 1 at 'index' in the row matrix, modulo 2"""

    # Check that matrix is a row matrix
    if len(row_matrix.shape) != 2 or row_matrix.shape[0] != 1:
        raise ValueError("The input should be a row matrix,"
                         f"not a {row_matrix.shape}-matrix")

    # If matrix[index] is not zero, put it to zero
    if index in row_matrix.indices:
        row_matrix.indices = np.setdiff1d(row_matrix.indices, [index])
    else:
        row_matrix.indices = np.append(row_matrix.indices, [index])

    row_matrix.data = np.ones(len(row_matrix.indices), dtype='uint8')
    row_matrix.indptr = np.array([0, len(row_matrix.indices)])


def vstack(matrices):
    return sparse.vstack(matrices, format='csr', dtype='uint8')


def hstack(matrices):
    return sparse.hstack(matrices, format='csr', dtype='uint8')


def hsplit(matrix):
    """Split a row matrix into two row matrices with half the number
    of columns"""

    if matrix.shape[1] % 2 != 0:
        raise ValueError("Matrix should have an even number of columns,"
                         f"not {matrix.shape[1]}")

    # Fast version for row matrices
    if matrix.shape[0] == 1:
        indices = np.array(matrix.indices)
        n = matrix.shape[1] // 2

        a_indices = indices[indices < n]
        b_indices = indices[indices >= n] - n

        a_indptr = np.array([0, len(a_indices)])
        b_indptr = np.array([0, len(b_indices)])

        a_data = np.ones(len(a_indices), dtype='uint8')
        b_data = np.ones(len(b_indices), dtype='uint8')

        a = csr_matrix((a_data, a_indices, a_indptr),
                       shape=(1, n), dtype='uint8')
        b = csr_matrix((b_data, b_indices, b_indptr),
                       shape=(1, n), dtype='uint8')
    else:
        n = matrix.shape[1] // 2
        a = matrix[:, :n]
        b = matrix[:, n:]

    return a, b


def dot(a, b):
    """Dot product between two row vectors a and b"""

    # Turn both inputs to sparse format (in case one is not)
    if not is_sparse(a):
        a = from_array(a)
    if not is_sparse(b):
        b = from_array(b)

    # Check if the dimensions are correct
    if a.shape[0] != 1 or b.shape[0] != 1:
        raise ValueError("Dot product only implemented for row vectors,"
                         f"not {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dimensions {a.shape} and {b.shape} don't agree"
                         "for dot product")

    n_common_ones = len(np.intersect1d(a.indices, b.indices))
    dot_product = int(n_common_ones % 2)

    return dot_product


def equal(a, b):
    """Test if two matrices are equal, or if a matrix equal a unique
    number everywhere"""

    # Matrix equality
    if isinstance(a, csr_matrix) and isinstance(b, csr_matrix):
        return np.all(a.shape == b.shape) and ((a != b).nnz == 0)

    # Matrix and number equality: a should be the int and b the matrix
    if isinstance(b, int) and isinstance(a, csr_matrix):
        a, b = b, a

    if isinstance(a, int) and isinstance(b, csr_matrix):
        if a == 0:
            return (b.nnz == 0)
        else:
            return (len(b.data) == np.product(b.shape)) and np.all(b.data == a)
    else:
        raise TypeError("Equality not supported between"
                        f"{type(a)} and {type(b)}")
