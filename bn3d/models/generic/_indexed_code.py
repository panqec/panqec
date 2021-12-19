from typing import Dict, Tuple, Optional
from abc import ABCMeta, abstractmethod
import numpy as np
from qecsim.model import StabilizerCode
from ...bpauli import bcommute
Indexer = Dict[Tuple[int, int, int], int]


class IndexedCode(StabilizerCode, metaclass=ABCMeta):
    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

    _size: np.ndarray
    _qubit_index: Indexer
    _vertex_index: Indexer
    _face_index: Indexer
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

        self._size = np.array([L_x, L_y, L_z])
        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()

    @property
    @abstractmethod
    def pauli_class(self):
        """The Pauli operator class."""

    @property
    def id(self):
        return self.__class__.__name__

    @property
    def qubit_index(self) -> Indexer:
        return self._qubit_index

    @property
    def vertex_index(self) -> Indexer:
        return self._vertex_index

    @property
    def face_index(self) -> Indexer:
        return self._face_index

    @property
    def n_faces(self) -> int:
        return len(self.face_index)

    @property
    def n_vertices(self) -> int:
        return len(self.vertex_index)

    @abstractmethod
    def _create_qubit_indices(self) -> Indexer:
        """Create qubit indices."""

    @abstractmethod
    def _create_vertex_indices(self) -> Indexer:
        """Create vertex indices."""

    @abstractmethod
    def _create_face_indices(self):
        """Create face indices."""

    @abstractmethod
    def axis(self, location) -> int:
        """ Return the axis corresponding to a given location"""

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
        if self._Hz.shape[0] == 0:
            self._Hz = self.stabilizers[:self.n_faces, :self.n_k_d[0]]
        return self._Hz

    @property
    def Hx(self) -> np.ndarray:
        if self._Hx.shape[0] == 0:
            self._Hx = self.stabilizers[self.n_faces:, self.n_k_d[0]:]
        return self._Hx

    def get_vertex_Z_stabilizers(self) -> np.ndarray:
        vertex_stabilizers = []

        for vertex_location in self.vertex_index.keys():
            operator = self.pauli_class(self)
            operator.vertex('Z', vertex_location)
            vertex_stabilizers.append(operator.to_bsf())

        return np.array(vertex_stabilizers, dtype=np.uint)

    def get_face_X_stabilizers(self) -> np.ndarray:
        face_stabilizers = []

        for face_location in self.face_index.keys():
            operator = self.pauli_class(self)
            operator.face('X', face_location)
            face_stabilizers.append(operator.to_bsf())

        return np.array(face_stabilizers, dtype=np.uint)

    def measure_syndrome(self, error) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizers, error.to_bsf())

    def is_face(self, location):
        return location in self.face_index.keys()

    def is_vertex(self, location):
        return location in self.vertex_index.keys()

    def is_qubit(self, location):
        return location in self.qubit_index.keys()
