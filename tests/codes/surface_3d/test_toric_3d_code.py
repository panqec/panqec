import numpy as np
import pytest
from panqec.bpauli import bsf_wt
from panqec.codes import Toric3DCode
import panqec.bsparse as bsparse
from tests.codes.stabilizer_code_test import StabilizerCodeTest


class TestToric3DCode(StabilizerCodeTest):

    @pytest.fixture(params=[(2, 2, 2), (3, 3, 3), (2, 3, 4)])
    def code(self, request):
        return Toric3DCode(*request.param)

    def test_get_vertex_stabilizers(self, code):
        n = code.n

        vertex_index = tuple(
            index
            for index, location in enumerate(code.stabilizer_coordinates)
            if code.stabilizer_type(location) == 'vertex'
        )
        stabilizers = code.stabilizer_matrix[vertex_index, :]

        # There should be least some vertex stabilizers.
        assert stabilizers.shape[0] > 0

        # For sparse matrices, shape[1] matters
        assert len(stabilizers.shape) == 1 or stabilizers.shape[1] > 0

        assert stabilizers.dtype == 'uint8'

        # All Z stabilizers should be weight 6.
        assert all(
            bsf_wt(stabilizer) == 6 for stabilizer in stabilizers
        )

        # Number of stabilizer generators should be number of vertices.
        assert stabilizers.shape[0] == np.product(code.size)

        # Each bsf length should be 2*n.
        assert stabilizers.shape[1] == 2*n

        # There should be no X or Y operators.
        if isinstance(stabilizers, np.ndarray):
            assert np.all(stabilizers[:, :n] == 0)
        else:
            assert bsparse.equal(stabilizers[:, :n], 0)

        assert all(
            'X' not in code.from_bsf(stabilizer)
            and 'Y' not in code.from_bsf(stabilizer)
            for stabilizer in stabilizers
        )

        # Each qubit should be in the support of exactly 2 stabilizers.
        assert np.all(stabilizers.sum(axis=0)[n:] == 2)

    def test_general_properties(self, code):
        n, k, d = code.n, code.k, code.d

        # The number of qubits should be the number of edges 3*L_x*L_y*L_z.
        assert n == 3*np.product(code.size)
        assert k == 3
        assert d == min(code.size)

    def test_get_face_stabilizers(self, code):
        n = code.n
        face_index = tuple(
            index
            for index, location in enumerate(code.stabilizer_coordinates)
            if code.stabilizer_type(location) == 'face'
        )
        stabilizers = code.stabilizer_matrix[face_index, :]

        # Weight of every stabilizer should be 4.
        assert np.all(stabilizers.sum(axis=1) == 4)
        assert stabilizers.dtype == 'uint8'

        # Number of stabilizer generators should be number of edges.
        assert stabilizers.shape[0] == 3*np.product(code.size)

        # The number of qubits should be the number of edges 3L^3.
        assert stabilizers.shape[1] == 2*n

        # There should be no Z or Y operators.
        if isinstance(stabilizers, np.ndarray):
            assert np.all(stabilizers[:, n:] == 0)
        else:
            assert bsparse.equal(stabilizers[:, n:], 0)

        # Each qubit should be in the support of exactly 4 stabilizers.
        if isinstance(stabilizers, np.ndarray):
            assert np.all(stabilizers.sum(axis=0)[:n] == 4)
        else:
            assert np.all(np.array(stabilizers.sum(axis=0)[0, :n]) == 4)

    def test_get_all_stabilizers(self, code):
        n = code.n
        n_vertex = len(code.type_index('vertex'))
        stabilizers = code.stabilizer_matrix

        # Total number of stabilizers.
        assert stabilizers.shape[0] == 4*np.product(code.size)

        # Z block of X stabilizers should be all 0.
        assert np.all(bsparse.to_array(stabilizers[:n_vertex, :n]) == 0)

        # X block of Z stabilizers should be all 0.
        assert np.all(
            bsparse.to_array((stabilizers[n_vertex:, n:])) == 0
        )

    def test_get_Z_logicals(self, code):
        n = code.n
        logicals = code.logicals_z
        assert logicals.shape[0] == 3
        assert logicals.shape[1] == 2*n

    def test_get_X_logicals(self, code):
        n = code.n
        logicals = code.logicals_x
        assert logicals.shape[0] == 3
        assert logicals.shape[1] == 2*n
