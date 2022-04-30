"""
Test PyMatching is working correctly for Toric Code as shown in their Tutorial.

These tests are largely copied from the tutorial provided by Pymatching.

:Author:
    Eric Huang
"""

import numpy as np
from scipy.sparse import hstack, kron, eye, csr_matrix, block_diag
from pymatching import Matching


def repetition_code(n):
    """Parity check matrix of a repetition code with length n. """
    row_ind, col_ind = zip(
        *((i, j) for i in range(n) for j in (i, (i + 1) % n))
    )
    data = np.ones(2*n, dtype=np.uint8)
    return csr_matrix((data, (row_ind, col_ind)))


def test_repetition_code():
    H = repetition_code(3)
    assert np.all(H.todense() == [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ])


def toric_code_x_stabilizers(L):
    """Sparse check matrix for the X stabilizers of a toric code with
    lattice size L, constructed as the hypergraph product of two repetition
    codes.
    """
    Hr = repetition_code(L)
    H = hstack(
            [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
            dtype=np.uint8
        )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csr_matrix(H)


def test_toric_code_x_stabilizers():
    """Test X stabilizers are ok."""
    L = 3
    H = toric_code_x_stabilizers(L)
    assert np.all(H.todense() == [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])


def toric_code_x_logicals(L):
    """Sparse binary matrix with each row corresponding to an X logical
    operator of a toric code with lattice size L. Constructed from the homology
    groups of the repetition codes using the Kunneth theorem.
    """
    H1 = csr_matrix(([1], ([0], [0])), shape=(1, L), dtype=np.uint8)
    H0 = csr_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csr_matrix(x_logicals)


def test_toric_code_x_locals():
    """Test logical operators are okay."""
    L = 3
    logicals = toric_code_x_logicals(L)
    assert np.all(logicals.todense() == [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    ])


def test_decode_toric_code():
    """Test decoding toric code."""
    L = 3
    p = 0.1
    np.random.seed(0)

    H = toric_code_x_stabilizers(L)
    logicals = toric_code_x_logicals(L)
    matching = Matching(H)

    n_trials = 100
    n_fails = 0

    for i in range(n_trials):
        noise = np.random.binomial(1, p, H.shape[1])
        syndrome = H.dot(noise) % 2
        correction = matching.decode(syndrome)
        total_error = (noise + correction) % 2
        logicals = toric_code_x_logicals(L)
        if np.any(total_error@logicals.T % 2):
            n_fails += 1

    assert n_fails/n_trials == 0.3
