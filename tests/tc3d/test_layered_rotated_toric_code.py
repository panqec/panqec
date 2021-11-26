from typing import Tuple, List
from abc import ABCMeta, abstractmethod
import pytest
from bn3d.tc3d import LayeredRotatedToricCode, LayeredToricPauli

from .indexed_code_test import IndexedCodeTest


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

    def test_every_face_operator_has_weight_4(self, code):
        for face in code.face_index:
            operator = LayeredToricPauli(code)
            operator.vertex('X', face)
            assert sum(operator.to_bsf()) == 4
