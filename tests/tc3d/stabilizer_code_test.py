import pytest
from abc import ABCMeta, abstractmethod
from panqec.codes import StabilizerCode


class StabilizerCodeTest(metaclass=ABCMeta):

    @pytest.fixture
    @abstractmethod
    def code(self) -> StabilizerCode:
        pass

    def test_n_equals_len_qubit_index(self, code):
        assert code.n == len(code.qubit_coordinates)

    def test_len_vertex_index_equals_number_of_vertex_stabilizers(self, code):
        n_vertices = len(code.vertex_index)
        assert n_vertices == code.get_vertex_stabilizers().shape[0]
        assert n_vertices == code.Hx.shape[0]

    def test_len_face_index_equals_number_of_face_stabilizers(self, code):
        n_faces = len(code.face_index)
        assert n_faces == code.get_face_stabilizers().shape[0]
        assert n_faces == code.Hz.shape[0]

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
