"""
Following the tutorial.

These tests are largely copied from the tutorial provided by Pymatching.

:Author:
    Eric Huang
"""
import pytest
from pymatching import Matching, set_seed
import networkx as nx
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


def test_spacetime_matching_graph(five_qubit_Hz):
    """Noisy stabilizer measurements."""

    Hz = five_qubit_Hz

    # Repetitions of stabilizer measurement rounds.
    repetitions = 5

    # Actual error probability.
    p = 0.05

    # Syndrome measurement error rate.
    q = 0.05

    m2d = Matching(
        Hz,
        spacelike_weights=np.log((1 - p)/p),
        repetitions=repetitions,
        timelike_weights=np.log((1 - p)/p),
    )

    num_stabilizers, num_qubits = Hz.shape
    np.random.seed(1)
    noise_new = (np.random.rand(num_qubits, repetitions) < p).astype(np.uint8)
    assert noise_new.dtype == np.uint8
    assert np.all(noise_new == [
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
    noise_total = noise_cumulative[:, -1]
    assert np.all(noise_cumulative == [
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    assert noise_cumulative.dtype == np.uint8

    # Noiseless syndrome.
    z_noiseless = Hz.dot(noise_cumulative) % 2
    assert np.all(z_noiseless == [
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ])

    # Syndrome errors.
    z_err = (np.random.rand(num_stabilizers, repetitions) < q).astype(np.uint8)
    z_err[:, -1] = 0
    assert np.all(z_err == [
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ])

    # Noisy syndromes.
    z_noisy = (z_noiseless + z_err) % 2
    assert np.all(z_noisy == [
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ])

    # Convert to difference syndrome.
    z_noisy[:, 1:] = (z_noisy[:, 1:] - z_noisy[:, 0:-1]) % 2
    assert np.all(z_noisy == [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ])

    # Decode.
    correction = m2d.decode(z_noisy)
    assert np.all(correction == [1, 0, 1, 0, 0])

    # Check cumulative total noise is all good.
    total_noise = (noise_total + correction) % 2
    assert np.all(total_noise == 0)


def test_loading_networkx_graphs():
    p = 0.2
    g = nx.Graph()

    g.add_edge(0, 1, qubit_id=0, weight=np.log((1 - p)/p), error_probability=p)
    g.add_edge(1, 2, qubit_id=1, weight=np.log((1 - p)/p), error_probability=p)
    g.add_edge(2, 3, qubit_id=2, weight=np.log((1 - p)/p), error_probability=p)
    g.add_edge(3, 4, qubit_id=3, weight=np.log((1 - p)/p), error_probability=p)
    g.add_edge(4, 5, qubit_id=4, weight=np.log((1 - p)/p), error_probability=p)

    # Single hook error on qubits 2 and 3
    p2 = 0.12
    g.add_edge(
        2, 4, qubit_id={2, 3}, weight=np.log((1 - p2)/p2), error_probability=p2
    )

    # Set optional attribute.
    g.nodes[0]['is_boundary'] = True
    g.nodes[5]['is_boundary'] = True

    # Connect boundary boundary with edge of weight 0.
    g.add_edge(0, 5, weight=0.0, qubit_id=-1, error_probability=0.0)

    m = Matching(g)
    assert isinstance(m, Matching)

    set_seed(1)
    noise, syndrome = m.add_noise()

    assert np.all(noise == [0, 1, 0, 0, 0])
    assert np.all(syndrome == [0, 1, 1, 0, 0, 0])

    # Decode.
    correction = m.decode(syndrome)

    # Test decoding worked.
    total_error = (correction + noise) % 2
    assert np.all(total_error == 0)
