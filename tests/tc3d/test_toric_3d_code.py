import numpy as np
import pytest
from bn3d.bpauli import bcommute, brank, bsf_wt, bsf_to_pauli
from bn3d.models import Toric3DCode
import bn3d.bsparse as bsparse


class TestToric3DCode:

    @pytest.fixture()
    def code(self):
        """Example code with non-uniform dimensions."""
        L_x, L_y, L_z = 5, 6, 7
        new_code = Toric3DCode(L_x, L_y, L_z)
        return new_code

    @pytest.mark.skip(reason='sparse')
    def test_cubic_code(self):
        code = Toric3DCode(5)
        assert np.all(code.size == [5, 5, 5])

    def test_get_vertex_stabilizers(self, code):
        n = code.n

        stabilizers = code.get_vertex_stabilizers()

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
            'X' not in bsf_to_pauli(stabilizer)
            and 'Y' not in bsf_to_pauli(stabilizer)
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
        stabilizers = code.get_face_stabilizers()

        # Weight of every stabilizer should be 6.
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
        stabilizers = code.stabilizers

        # Total number of stabilizers.
        assert stabilizers.shape[0] == 4*np.product(code.size)

        # Z block of X stabilizers should be all 0.
        assert np.all(bsparse.to_array(stabilizers[:n, n:]) == 0)

        # X block of Z stabilizers should be all 0.
        assert np.all(bsparse.to_array((stabilizers[n:, :np.product(code.size)])) == 0)

    def test_get_Z_logicals(self, code):
        n = code.n
        logicals = code.logical_zs
        assert logicals.shape[0] == 3
        assert logicals.shape[1] == 2*n

    def test_get_X_logicals(self, code):
        n = code.n
        logicals = code.logical_xs
        assert logicals.shape[0] == 3
        assert logicals.shape[1] == 2*n


class TestCommutationRelationsToric3DCode:

    @pytest.fixture()
    def code(self):
        """Example code with non-uniform dimensions."""
        L_x, L_y, L_z = 3, 4, 5
        new_code = Toric3DCode(L_x, L_y, L_z)
        return new_code

    def test_stabilizers_commute_with_each_other(self, code):
        assert np.all(bcommute(code.stabilizers, code.stabilizers) == 0)

    def test_Z_logicals_commute_with_each_other(self, code):
        assert np.all(bcommute(code.logical_zs, code.logical_zs) == 0)

    def test_X_logicals_commute_with_each_other(self, code):
        assert np.all(bcommute(code.logical_xs, code.logical_xs) == 0)

    def test_stabilizers_commute_with_logicals(self, code):
        if isinstance(code.logical_xs, np.ndarray):
            logicals = np.concatenate([code.logical_xs, code.logical_zs])
        else:
            logicals = bsparse.vstack([code.logical_xs, code.logical_zs])
        assert np.all(bcommute(logicals, code.stabilizers) == 0)

    def test_logicals_anticommute_correctly(self, code):
        assert np.all(
            bcommute(code.logical_xs, code.logical_zs) == np.eye(code.k)
        )
