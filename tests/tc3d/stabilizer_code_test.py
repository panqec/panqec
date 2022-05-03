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
        if 'vertex' in code.stabilizer_types and code.is_css:
            n_vertices = len(code.type_index('vertex'))
            assert n_vertices == code.Hz.shape[0]

    def test_len_face_index_equals_number_of_face_stabilizers(self, code):
        if 'face' in code.stabilizer_types and code.is_css:
            n_faces = len(code.type_index('face'))
            assert n_faces == code.Hx.shape[0]

    def test_qubit_index(self, code):
        assert all(len(index) == code.dimension for index in code.qubit_index)

    def test_stabilizer_index(self, code):
        assert all(
            len(index) == code.dimension for index in code.stabilizer_index
        )

    def test_qubit_stabilizer_indices_no_overlap(self, code):
        qubits = set(code.qubit_index.keys())
        stabilizers = set(code.stabilizer_index.keys())
        assert qubits.isdisjoint(stabilizers)
