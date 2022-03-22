from typing import Dict, Tuple, Optional
from abc import ABCMeta
from ._indexed_code import IndexedCode
from ... import bsparse

Indexer = Dict[Tuple[int, int, int], int]


class IndexedSparseCode(IndexedCode, metaclass=ABCMeta):
    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
        deformed_axis: Optional[int] = None
    ):
        if L_y is None:
            L_y = L_x
        if L_z is None:
            L_z = L_x

        self._deformed_axis = deformed_axis

        if self.dimension == 2:
            self._size = (L_x, L_y)
        else:
            self._size = (L_x, L_y, L_z)

        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()

        self._stabilizers = bsparse.empty_row(2*self.n)
        self._Hx = bsparse.empty_row(self.n)
        self._Hz = bsparse.empty_row(self.n)
        self._logical_xs = bsparse.empty_row(self.n)
        self._logical_zs = bsparse.empty_row(self.n)

    @property
    def stabilizers(self):
        if bsparse.is_empty(self._stabilizers):
            face_stabilizers = self.get_face_X_stabilizers()
            vertex_stabilizers = self.get_vertex_Z_stabilizers()
            self._stabilizers = bsparse.vstack([
                face_stabilizers,
                vertex_stabilizers,
            ])
        return self._stabilizers

    def get_vertex_Z_stabilizers(self):
        vertex_stabilizers = bsparse.empty_row(2*self.n)

        for vertex_location in self.vertex_index.keys():
            operator = self.pauli_class(self)
            operator.vertex('Z', vertex_location, deformed_axis=self._deformed_axis)
            vertex_stabilizers = bsparse.vstack([vertex_stabilizers, operator.to_bsf()])

        return vertex_stabilizers

    def get_face_X_stabilizers(self):
        face_stabilizers = bsparse.empty_row(2*self.n)

        for face_location in self.face_index.keys():
            operator = self.pauli_class(self)
            operator.face('X', face_location, deformed_axis=self._deformed_axis)
            face_stabilizers = bsparse.vstack([face_stabilizers, operator.to_bsf()])

        return face_stabilizers
