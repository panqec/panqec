"""
Following the tutorial.
"""
import pytest
from pymatching import Matching
import numpy as np
import scipy


@pytest.fixture
def five_qubit_Hz():
    """Five-qubit bit flip code parity check matrix."""
    Hz = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ])
    return Hz


def test_getting_started(five_qubit_Hz):
    """From the tutorial."""

    Hz = five_qubit_Hz
    m = Matching(Hz)

    # Noise example IIXXI.
    noise = np.array([0, 0, 1, 1, 0])

    # Syndrome vector.
    z = Hz.dot(noise) % 2
    assert np.all(z == [0, 1, 0, 1])

    # Decode the syndrome.
    c = m.decode(z)

    assert type(c) is np.ndarray
    assert np.all(c == [0, 0, 1, 1, 0])


def test_getting_started_sparse(five_qubit_Hz):
    """Sparse version."""

    # Sparse version.
    Hz = scipy.sparse.csr_matrix(five_qubit_Hz)

    m = Matching(Hz)

    # Noise example IIXXI.
    noise = np.array([0, 0, 1, 1, 0])

    # Syndrome vector.
    z = Hz.dot(noise) % 2
    assert np.all(z == [0, 1, 0, 1])

    # Decode the syndrome.
    c = m.decode(z)

    assert type(c) is np.ndarray
    assert np.all(c == [0, 0, 1, 1, 0])
