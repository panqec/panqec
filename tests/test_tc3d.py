import pytest
import numpy as np
from panqec.bpauli import bcommute, get_effective_error
from pymatching import Matching
from panqec.codes import Toric3DCode


@pytest.fixture
def code(L):
    return Toric3DCode(L, L, L)


@pytest.fixture
def L():
    return 3


class TestStabilizerLogicalCounts:
    def test_get_vertex_stabilizers(self, code, L):
        stabilizers = np.array([
            code.stabilizer_matrix[index].toarray()[0].tolist()
            for index, location in enumerate(code.stabilizer_coordinates)
            if code.stabilizer_type(location) == 'vertex'
        ], dtype=np.uint)
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

    def test_get_face_stabilizers(self, code, L):
        stabilizers = np.array([
            code.stabilizer_matrix[index].toarray()[0].tolist()
            for index, location in enumerate(code.stabilizer_coordinates)
            if code.stabilizer_type(location) == 'face'
        ], dtype=np.uint)

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

    def test_get_all_stabilizers(self, code, L):
        stabilizers = np.array(
            code.stabilizer_matrix.toarray().tolist(),
            dtype=np.uint
        )

        # Total number of stabilizers.
        n_vertices = L**3
        n_faces = 3*L**3
        assert stabilizers.shape[0] == n_vertices + n_faces

        # Number of qubits.
        n_qubits = 3*L**3
        assert stabilizers.shape[1] == 2*n_qubits

        # Z block of X face stabilizers should be all 0.
        assert np.all(stabilizers[n_vertices:, n_qubits:] == 0)

        # X block of Z vertex stabilizers should be all 0.
        assert np.all(stabilizers[:n_vertices, :n_qubits] == 0)

    def test_get_Z_logicals(self, code, L):
        logicals = code.logicals_z
        assert logicals.shape[0] == 3
        assert logicals.shape[1] == 2*3*L**3

    def test_get_X_logicals(self, code, L):
        logicals = code.logicals_x
        assert logicals.shape[0] == 3
        assert logicals.shape[1] == 2*3*L**3


class TestCommutationRelations:

    def test_stabilizers_commute_with_each_other(self, code):
        stabilizers = code.stabilizer_matrix.toarray()
        assert np.all(bcommute(stabilizers, stabilizers) == 0)

    def test_Z_logicals_commute_with_each_other(self, code):
        logicals = code.logicals_z
        assert np.all(bcommute(logicals, logicals) == 0)

    def test_X_logicals_commute_with_each_other(self, code):
        logicals = code.logicals_x
        assert np.all(bcommute(logicals, logicals) == 0)

    def test_stabilizers_commute_with_logicals(self, code):
        stabilizers = code.stabilizer_matrix.toarray()
        logicals = np.vstack([code.logicals_x, code.logicals_z])
        assert np.all(bcommute(logicals, stabilizers) == 0)

    def test_X_and_Z_logicals_commutation(self, code, L):
        X_logicals = code.logicals_x
        Z_logicals = code.logicals_z
        commutation = bcommute(X_logicals, Z_logicals)
        assert np.all(commutation == np.identity(L))


def test_correcting_X_noise_produces_X_logical_errors_only(code, L):
    p = 0.5
    np.random.seed(0)
    Z_stabilizers = np.array([
        code.stabilizer_matrix[index].toarray()[0].tolist()
        for index, location in enumerate(code.stabilizer_coordinates)
        if code.stabilizer_type(location) == 'vertex'
    ], dtype=np.uint)
    H_Z = Z_stabilizers[:, 3*L**3:]
    X_logicals = code.logicals_x
    Z_logicals = code.logicals_z

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
