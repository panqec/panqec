import pytest
import numpy as np
from bn3d.tc3d import RotatedToricCode3D


class TestRotatedToricCode3D:

    L_x = 2
    L_y = 3
    L_z = 4

    @pytest.fixture
    def code(self):
        """Example code with non-uniform dimensions."""
        new_code = RotatedToricCode3D(self.L_x, self.L_y, self.L_z)
        return new_code

    def test_n_k_d(self, code):
        n_horizontals = 2*self.L_y*(2*self.L_x + 1)*self.L_z
        n_verticals = (2*self.L_x*self.L_y)*self.L_z
        assert n_horizontals + n_verticals == code.n_k_d[0]

    def test_qubit_index(self, code):
        assert all(len(index) == 3 for index in code.qubit_index)
        assert sorted(code.qubit_index.values()) == sorted(
            range(len(code.qubit_index))
        )

    def test_vertex_index(self, code):
        assert all(len(index) == 3 for index in code.vertex_index)
        assert sorted(code.vertex_index.values()) == sorted(
            range(len(code.vertex_index))
        )

    def test_face_index(self, code):
        assert all(len(index) == 3 for index in code.face_index)
        assert sorted(code.face_index.values()) == sorted(
            range(len(code.face_index))
        )

    def test_qubit_vertex_face_indices_no_overlap(self, code):
        qubits = set(code.qubit_index.keys())
        vertices = set(code.vertex_index.keys())
        faces = set(code.face_index.keys())
        assert qubits.isdisjoint(vertices)
        assert qubits.isdisjoint(faces)
        assert vertices.isdisjoint(faces)

    def test_vertex_index_corner_region(self, code):

        # First layer corner near origin
        assert (0, 0, 0) not in code.vertex_index
        assert (2, 0, 1) in code.vertex_index
        assert (2, 4, 1) in code.vertex_index
        assert (0, 2, 1) not in code.vertex_index
        assert (4, 2, 1) in code.vertex_index
        assert (6, 0, 1) in code.vertex_index
        assert (2, 8, 1) in code.vertex_index
        assert (4, 6, 1) in code.vertex_index

    def test_vertex_index_complies_with_rules(self, code):
        for x, y, z in code.vertex_index.keys():
            assert z % 2 == 1
            if x % 4 == 2:
                assert y % 4 == 0
            else:
                assert x % 4 == 0
                assert y % 4 == 2

    def test_vertex_index_boundary_conditions(self, code):
        for x, y, z in code.vertex_index.keys():
            assert x != 0
            if y == 0:
                assert x % 4 == 2
            if self.L_x % 2 == 0:
                if x == self.L_x*2 + 2:
                    assert y % 4 == 0

    def test_face_index_complies_with_rules(self, code):
        for x, y, z in code.face_index.keys():
            if z % 2 == 1:
                if x % 4 == 0:
                    assert y % 4 == 0
                else:
                    assert x % 4 == 2
                    assert y % 4 == 2
            else:
                assert x % 2 == 1
                assert y % 2 == 1

    def test_qubit_index_complies_with_rules(self, code):
        for x, y, z in code.qubit_index.keys():
            if z % 2 == 1:
                assert x % 2 == 1
                assert y % 2 == 1
            else:
                if x % 4 == 0:
                    assert y % 4 == 2
                else:
                    assert x % 4 == 2
                    assert y % 4 == 0

    def test_each_qubit_contained_in_1_or_2_check_operators(self, code):
        H = code.Hx
        assert np.all(H.sum(axis=0) > 0)
        assert np.all(H.sum(axis=0) <= 2)
