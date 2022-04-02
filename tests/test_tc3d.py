import numpy as np
from panqec.bpauli import bcommute, get_effective_error
from pymatching import Matching
from panqec.codes import (
    get_vertex_stabilizers, get_face_stabilizers, get_all_stabilizers,
    get_Z_logicals, get_X_logicals, get_all_logicals,
)


def test_get_vertex_stabilizers():
    L = 3
    stabilizers = get_vertex_stabilizers(L)
    assert stabilizers.dtype == np.uint

    # Weight of every stabilizer should be 6.
    assert np.all(stabilizers.sum(axis=1) == 6)

    # Number of stabilizer generators should be number of vertices.
    assert stabilizers.shape[0] == L**3

    # The number of qubits should be the number of edges 3L^3.
    assert stabilizers.shape[1] == 2*3*L**3

    # There should be no X or Y operators.
    assert np.all(stabilizers[:, :3*L**3] == 0)

    # Each qubit should be in the support of exactly 2 stabilizers.
    assert np.all(stabilizers.sum(axis=0)[3*L**3:] == 2)


def test_get_face_stabilizers():
    L = 3
    stabilizers = get_face_stabilizers(L)

    # Weight of every stabilizer should be 6.
    assert np.all(stabilizers.sum(axis=1) == 4)
    assert stabilizers.dtype == np.uint

    # Number of stabilizer generators should be number of edges.
    assert stabilizers.shape[0] == 3*L**3

    # The number of qubits should be the number of edges 3L^3.
    assert stabilizers.shape[1] == 2*3*L**3

    # There should be no Z or Y operators.
    assert np.all(stabilizers[:, 3*L**3:] == 0)

    # Each qubit should be in the support of exactly 4 stabilizers.
    assert np.all(stabilizers.sum(axis=0)[:3*L**3] == 4)


def test_get_all_stabilizers():
    L = 3
    stabilizers = get_all_stabilizers(3)

    # Total number of stabilizers.
    assert stabilizers.shape[0] == 4*3*L**2
    # Z block of X stabilizers should be all 0.
    assert np.all(stabilizers[:3*L**3, 3*L**3:] == 0)

    # X block of Z stabilizers should be all 0.
    assert np.all(stabilizers[3*L**3:, :L**3] == 0)


def test_get_Z_logicals():
    L = 3
    logicals = get_Z_logicals(L)
    assert logicals.shape[0] == 3
    assert logicals.shape[1] == 2*3*L**3


def test_get_X_logicals():
    L = 3
    logicals = get_X_logicals(L)
    assert logicals.shape[0] == 3
    assert logicals.shape[1] == 2*3*L**3


class TestCommutationRelations:

    def test_stabilizers_commute_with_each_other(self):
        L = 3
        stabilizers = get_all_stabilizers(L)
        assert np.all(bcommute(stabilizers, stabilizers) == 0)

    def test_Z_logicals_commute_with_each_other(self):
        L = 3
        logicals = get_Z_logicals(L)
        assert np.all(bcommute(logicals, logicals) == 0)

    def test_X_logicals_commute_with_each_other(self):
        L = 3
        logicals = get_X_logicals(L)
        assert np.all(bcommute(logicals, logicals) == 0)

    def test_stabilizers_commute_with_logicals(self):
        L = 3
        stabilizers = get_all_stabilizers(L)
        logicals = get_all_logicals(L)
        assert np.all(bcommute(logicals, stabilizers) == 0)

    def test_X_and_Z_logicals_commutation(self):
        L = 3
        X_logicals = get_X_logicals(L)
        Z_logicals = get_Z_logicals(L)
        commutation = bcommute(X_logicals, Z_logicals)
        assert np.all(commutation == np.identity(L))


def test_correcting_X_noise_produces_X_logical_errors_only():
    L = 3
    p = 0.5
    np.random.seed(0)
    Z_stabilizers = get_vertex_stabilizers(L)
    H_Z = Z_stabilizers[:, 3*L**3:]
    X_logicals = get_X_logicals(L)
    Z_logicals = get_Z_logicals(L)

    matching = Matching(H_Z)
    noise_X = np.random.binomial(1, p, H_Z.shape[1])

    # Make sure the noise is non-trivial.
    assert noise_X.sum() > 0
    syndrome_Z = H_Z.dot(noise_X) % 2
    correction_X = matching.decode(syndrome_Z, num_neighbours=None)
    total_error_X = (noise_X + correction_X) % 2
    total_error = np.zeros(2*3*L**3, dtype=np.uint)
    total_error[:3*L**3] = total_error_X

    # Compute the effective error on the logical qubits.
    effective_error = get_effective_error(
        total_error, X_logicals, Z_logicals
    )

    # Make sure the total error is non-trivial.
    assert total_error.sum() > 0

    # Assert that the logical error is non-trivial.
    assert np.any(effective_error != 0)

    # Therefore it should anticommute with some Z logicals.
    assert np.any(bcommute(Z_logicals, total_error) == 1)

    # Total error is in code space.
    assert np.all(bcommute(Z_stabilizers, total_error) == 0)

    # Total commutes with all X logicals.
    assert np.all(bcommute(X_logicals, total_error) == 0)
