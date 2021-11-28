from typing import Tuple, List
from abc import ABCMeta, abstractmethod
import pytest
import numpy as np
from bn3d.tc3d import LayeredRotatedToricCode, LayeredToricPauli
from bn3d.bpauli import bcommute, bvector_to_pauli_string, brank

from .indexed_code_test import IndexedCodeTest


def operator_spec(code, bsf):
    """Get representation of BSF as list of (pauli, (x, y, z)) entries.

    Useful for debugging and reading BSF in human-readable format.
    """
    operator_spec = []
    pauli_string = bvector_to_pauli_string(bsf)
    for index, pauli in enumerate(pauli_string):
        if pauli != 'I':
            matches = [
                xyz
                for xyz, i in code.qubit_index.items()
                if i == index
            ]
            location = matches[0]
            operator_spec.append((pauli, location))
    return operator_spec


class IndexedCodeTestWithCoordinates(IndexedCodeTest, metaclass=ABCMeta):

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int, int]:
        """Plane size x"""

    @property
    @abstractmethod
    def expected_plane_edges_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane edges in unrotated lattice."""

    @property
    @abstractmethod
    def expected_plane_faces_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane faces in unrotated lattice."""

    @property
    @abstractmethod
    def expected_plane_vertices_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane vertices in unrot lattice."""

    @property
    @abstractmethod
    def expected_plane_z(self) -> List[int]:
        """Expected z coordinates of horizontal planes."""

    @property
    @abstractmethod
    def expected_vertical_z(self) -> List[int]:
        """Expected z coordinates of vertical edges and faces."""

    @pytest.fixture
    def code(self):
        return LayeredRotatedToricCode(*self.size)

    def test_qubit_indices(self, code):
        locations = set(code.qubit_index.keys())
        expected_locations = set([
            (x, y, z)
            for x, y in self.expected_plane_edges_xy
            for z in self.expected_plane_z
        ] + [
            (x, y, z)
            for x, y in self.expected_plane_vertices_xy
            for z in self.expected_vertical_z
        ])
        assert locations == expected_locations

    def test_vertex_indices(self, code):
        locations = set(code.vertex_index.keys())
        expected_locations = set([
            (x, y, z)
            for x, y in self.expected_plane_vertices_xy
            for z in self.expected_plane_z
        ])
        assert locations == expected_locations

    def test_face_indices(self, code):
        locations = set(code.face_index.keys())
        expected_locations = set([
            (x, y, z)
            for x, y in self.expected_plane_edges_xy
            for z in self.expected_vertical_z
        ] + [
            (x, y, z)
            for x, y in self.expected_plane_faces_xy
            for z in self.expected_plane_z
        ])
        assert locations == expected_locations

    def test_all_stabilizers_commute(self, code):
        commutators = bcommute(code.stabilizers, code.stabilizers)
        non_commuting = set([
            (i, j)
            for i, j in np.array(np.where(commutators)).T
            if i <= j
        ])

        # Print the first few non-commuting stabilizers if any found.
        if non_commuting:
            max_print = 5
            for i_print, (i, j) in enumerate(non_commuting):
                print(f'Stabilizers {i} and {j} anticommuting')
                print(f'{i}:', operator_spec(code, code.stabilizers[i]))
                print(f'{j}:', operator_spec(code, code.stabilizers[j]))
                if i_print == max_print:
                    break

        # There should be no non-commuting pairs of stabilizers.
        assert len(non_commuting) == 0

    def test_n_indepdent_stabilizers_equals_n_minus_k(self, code):
        n, k, _ = code.n_k_d
        matrix = code.stabilizers

        # Number of independent stabilizer generators.
        rank = brank(matrix)

        assert rank <= matrix.shape[0]
        assert rank == n - k


class TestLayeredRotatedToricCode2x2x1(IndexedCodeTestWithCoordinates):
    size = (2, 2, 1)
    expected_plane_edges_xy = [
        (1, 1), (3, 1),
        (1, 3), (3, 3),
    ]
    expected_plane_faces_xy = [
        (2, 2), (4, 4)
    ]
    expected_plane_vertices_xy = [
        (4, 2), (2, 4)
    ]
    expected_plane_z = [1, 3]
    expected_vertical_z = [2]


class TestLayeredRotatedToricCode3x2x1(IndexedCodeTestWithCoordinates):
    size = (3, 2, 1)
    expected_plane_edges_xy = [
        (1, 1), (3, 1), (5, 1),
        (1, 3), (3, 3), (5, 3),
    ]
    expected_plane_faces_xy = [
        (2, 2), (4, 4), (6, 2),
    ]
    expected_plane_vertices_xy = [
        (4, 2), (2, 4), (6, 4)
    ]
    expected_plane_z = [1, 3]
    expected_vertical_z = [2]


@pytest.mark.skip(reason='odd by odd')
class TestLayeredRotatedToricCode3x3x3(IndexedCodeTestWithCoordinates):
    size = (3, 3, 3)
    expected_plane_edges_xy = [
        (1, 1), (3, 1), (5, 1),
        (1, 3), (3, 3), (5, 3),
        (1, 5), (3, 5), (5, 5),
    ]
    expected_plane_faces_xy = [
        (2, 2), (6, 2),
        (4, 4),
        (2, 6), (6, 6),
    ]
    expected_plane_vertices_xy = [
        (4, 2),
        (2, 4), (6, 4),
        (4, 6),
    ]
    expected_plane_z = [1, 3, 5, 7]
    expected_vertical_z = [2, 4, 6]


class TestLayeredRotatedToricCode4x3x3(IndexedCodeTestWithCoordinates):
    size = (4, 3, 3)
    expected_plane_edges_xy = [
        (1, 1), (3, 1), (5, 1), (7, 1),
        (1, 3), (3, 3), (5, 3), (7, 3),
        (1, 5), (3, 5), (5, 5), (7, 5),
    ]
    expected_plane_faces_xy = [
        (2, 2), (6, 2),
        (4, 4), (8, 4),
        (2, 6), (6, 6),
    ]
    expected_plane_vertices_xy = [
        (4, 2), (8, 2),
        (2, 4), (6, 4),
        (4, 6), (8, 6),
    ]
    expected_plane_z = [1, 3, 5, 7]
    expected_vertical_z = [2, 4, 6]


class TestLayeredRotatedToricCode3x4x3(IndexedCodeTestWithCoordinates):
    size = (3, 4, 3)
    expected_plane_edges_xy = [
        (1, 1), (1, 3), (1, 5), (1, 7),
        (3, 1), (3, 3), (3, 5), (3, 7),
        (5, 1), (5, 3), (5, 5), (5, 7),
    ]
    expected_plane_faces_xy = [
        (2, 2), (2, 6),
        (4, 4), (4, 8),
        (6, 2), (6, 6),
    ]
    expected_plane_vertices_xy = [
        (2, 4), (2, 8),
        (4, 2), (4, 6),
        (6, 4), (6, 8),
    ]
    expected_plane_z = [1, 3, 5, 7]
    expected_vertical_z = [2, 4, 6]


class TestLayeredRotatedToricCode4x4x3(IndexedCodeTestWithCoordinates):
    size = (4, 4, 3)
    expected_plane_edges_xy = [
        (1, 1), (1, 3), (1, 5), (1, 7),
        (3, 1), (3, 3), (3, 5), (3, 7),
        (5, 1), (5, 3), (5, 5), (5, 7),
        (7, 1), (7, 3), (7, 5), (7, 7),
    ]
    expected_plane_faces_xy = [
        (2, 2), (2, 6),
        (4, 4), (4, 8),
        (6, 2), (6, 6),
        (8, 4), (8, 8),
    ]
    expected_plane_vertices_xy = [
        (2, 4), (2, 8),
        (4, 2), (4, 6),
        (6, 4), (6, 8),
        (8, 2), (8, 6),
    ]
    expected_plane_z = [1, 3, 5, 7]
    expected_vertical_z = [2, 4, 6]


class TestLayeredRotatedToricPauli:

    size = (4, 3, 3)

    @pytest.fixture
    def code(self):
        """Example code with co-prime x and y dimensions."""
        return LayeredRotatedToricCode(*self.size)

    def test_vertex_operator_in_bulk_has_weight_6(self, code):
        vertices = [
            (x, y, z)
            for x, y, z in code.vertex_index
            if z in [3, 5]
        ]
        for vertex in vertices:
            operator = LayeredToricPauli(code)
            operator.vertex('Z', vertex)
            assert sum(operator.to_bsf()) == 6

    def test_vertex_operator_on_boundary_has_weight_5(self, code):
        vertices = [
            (x, y, z)
            for x, y, z in code.vertex_index
            if z in [1, 7]
        ]
        for vertex in vertices:
            operator = LayeredToricPauli(code)
            operator.vertex('Z', vertex)
            assert sum(operator.to_bsf()) == 5

    def test_every_face_operator_in_bulk_has_weight_4(self, code):
        for face in code.face_index:
            x, y, z = face
            if x > 1 and y > 1 and z > 1 and z < 2*self.size[2] + 1:
                operator = LayeredToricPauli(code)
                operator.face('X', face)
                assert sum(operator.to_bsf()) == 4
