from typing import Dict, Tuple, Optional
from abc import ABCMeta, abstractmethod
import numpy as np
from qecsim.model import StabilizerCode
from bn3d.bpauli import bcommute
Indexer = Dict[Tuple[int, int, int], int]


class IndexedCode(StabilizerCode, metaclass=ABCMeta):

    _size: Tuple[int, int, int]
    _qubit_index: Dict[Tuple[int, int, int], int]
    _vertex_index: Dict[Tuple[int, int, int], int]
    _face_index: Dict[Tuple[int, int, int], int]
    _stabilizers = np.array([])
    _Hx = np.array([])
    _Hz = np.array([])
    _logical_xs = np.array([])
    _logical_zs = np.array([])

    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None
    ):
        if L_y is None:
            L_y = L_x
        if L_z is None:
            L_z = L_x

        self._size = (L_x, L_y, L_z)
        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()

    @property
    def qubit_index(self) -> Indexer:
        return self._qubit_index

    @property
    def vertex_index(self) -> Indexer:
        return self._vertex_index

    @property
    def face_index(self) -> Indexer:
        return self._face_index

    @abstractmethod
    def _create_vertex_indices(self) -> Indexer:
        """Create vertex indices."""

    @abstractmethod
    def _create_face_indices(self):
        """Create face indices."""

    @property
    def stabilizers(self) -> np.ndarray:
        if self._stabilizers.size == 0:
            face_stabilizers = self.get_face_X_stabilizers()
            vertex_stabilizers = self.get_vertex_Z_stabilizers()
            self._stabilizers = np.concatenate([
                face_stabilizers,
                vertex_stabilizers,
            ])
        return self._stabilizers

    @property
    def size(self) -> Tuple[int, int, int]:
        """Dimensions of lattice."""
        return self._size

    @property
    def Hz(self) -> np.ndarray:
        if self._Hz.size == 0:
            self._Hz = self.get_face_X_stabilizers()
        return self._Hz[:, :self.n_k_d[0]]

    @property
    def Hx(self) -> np.ndarray:
        if self._Hx.size == 0:
            self._Hx = self.get_vertex_Z_stabilizers()
        return self._Hx[:, self.n_k_d[0]:]

    def get_vertex_Z_stabilizers(self) -> np.ndarray:
        vertex_stabilizers = []

        for (x, y, z) in self.vertex_index.keys():
            operator = self.pauli_class(self)
            operator.vertex('Z', (x, y, z))
            vertex_stabilizers.append(operator.to_bsf())

        return np.array(vertex_stabilizers, dtype=np.uint)

    def get_face_X_stabilizers(self) -> np.ndarray:
        face_stabilizers = []

        for (x, y, z) in self.face_index.keys():
            operator = self.pauli_class(self)
            operator.face('X', (x, y, z))
            face_stabilizers.append(operator.to_bsf())

        return np.array(face_stabilizers, dtype=np.uint)

    def measure_syndrome(self, error) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizers, error.to_bsf())
