import pytest
from bn3d.tc3d import RotatedToricCode3D


class TestRotatedToricCode3D:

    @pytest.fixture
    def code(self):
        """Example code with non-uniform dimensions."""
        L_x, L_y, L_z = 2, 3, 4
        new_code = RotatedToricCode3D(L_x, L_y, L_z)
        return new_code

    def test_n_k_d(self, code):
        L_x, L_y, L_z = 2, 3, 4
        n_horizontals = 2*L_y*(2*L_x + 1)*L_z
        n_verticals = (2*L_x*L_y)*L_z
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
