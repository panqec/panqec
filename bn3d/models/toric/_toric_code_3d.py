import itertools
from typing import Tuple, Optional, Dict
import numpy as np
from qecsim.model import StabilizerCode
from ._toric_3d_pauli import Toric3DPauli
from ...bpauli import bcommute


class ToricCode3D(StabilizerCode):

    _shape: Tuple[int, int, int, int]
    _size: Tuple[int, int, int]
    _qubit_index: Dict[Tuple[int, int, int, int], int]
    _vertex_index: Dict[Tuple[int, int, int], int]
    _face_index: Dict[Tuple[int, int, int, int], int]
    X_AXIS: int = 0
    Y_AXIS: int = 1
    Z_AXIS: int = 2
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
        self._shape = (3, L_x, L_y, L_z)
        self._size = (L_x, L_y, L_z)
        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()

    # StabilizerCode interface methods.

    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        return (np.product(self.shape), 3, min(self.size))

    @property
    def qubit_index(self) -> Dict[Tuple[int, int, int, int], int]:
        return self._qubit_index

    @property
    def vertex_index(self) -> Dict[Tuple[int, int, int], int]:
        return self._vertex_index

    @property
    def face_index(self) -> Dict[Tuple[int, int, int, int], int]:
        return self._face_index
    
    @property
    def n_faces(self) -> int:
        return len(self.face_index)

    @property
    def n_vertices(self) -> int:
        return len(self.vertex_index)

    @property
    def label(self) -> str:
        return 'Toric {}x{}x{}'.format(*self.size)

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
    def Hz(self) -> np.ndarray:
        if self._Hz.size == 0:
            self._Hz = self.stabilizers[:self.n_faces, :self.n_k_d[0]]
        return self._Hz

    @property
    def Hx(self) -> np.ndarray:
        if self._Hx.size == 0:
            self._Hx = self.stabilizers[self.n_faces:, self.n_k_d[0]:]
        return self._Hx

    @property
    def logical_xs(self) -> np.ndarray:
        """The 3 logical X operators."""

        if self._logical_xs.size == 0:
            L_x, L_y, L_z = self.size
            logicals = []

            # X operators along x edges in x direction.
            logical = Toric3DPauli(self)
            for x in range(L_x):
                logical.site('X', (0, x, 0, 0))
            logicals.append(logical.to_bsf())

            # X operators along y edges in y direction.
            logical = Toric3DPauli(self)
            for y in range(L_y):
                logical.site('X', (1, 0, y, 0))
            logicals.append(logical.to_bsf())

            # X operators along z edges in z direction
            logical = Toric3DPauli(self)
            for z in range(L_z):
                logical.site('X', (2, 0, 0, z))
            logicals.append(logical.to_bsf())

            self._logical_xs = np.array(logicals, dtype=np.uint)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the 3 logical Z operators."""
        if self._logical_zs.size == 0:
            L_x, L_y, L_z = self.size
            logicals = []

            # Z operators on x edges forming surface normal to x (yz plane).
            logical = Toric3DPauli(self)
            for y in range(L_y):
                for z in range(L_z):
                    logical.site('Z', (0, 0, y, z))
            logicals.append(logical.to_bsf())

            # Z operators on y edges forming surface normal to y (zx plane).
            logical = Toric3DPauli(self)
            for z in range(L_z):
                for x in range(L_x):
                    logical.site('Z', (1, x, 0, z))
            logicals.append(logical.to_bsf())

            # Z operators on z edges forming surface normal to z (xy plane).
            logical = Toric3DPauli(self)
            for x in range(L_x):
                for y in range(L_y):
                    logical.site('Z', (2, x, y, 0))
            logicals.append(logical.to_bsf())

            self._logical_zs = np.array(logicals, dtype=np.uint)

        return self._logical_zs

    @property
    def size(self) -> Tuple[int, int, int]:
        """Dimensions of lattice."""
        return self._shape[1:]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Shape of lattice for each qubit."""
        return self._shape

    def _create_qubit_indices(self):
        ranges = [range(length) for length in self.shape]
        coordinates = [
            (axis, x, y, z) for axis, x, y, z in itertools.product(*ranges)
        ]

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self):
        ranges = [range(length) for length in self.size]
        coordinates = [(x, y, z) for x, y, z in itertools.product(*ranges)]

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        ranges = [range(length) for length in self.shape]
        coordinates = [
            (axis, x, y, z) for axis, x, y, z in itertools.product(*ranges)
        ]

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def get_vertex_Z_stabilizers(self) -> np.ndarray:
        vertex_stabilizers = []

        for (x, y, z) in self.vertex_index.keys():
            operator = Toric3DPauli(self)
            operator.vertex('Z', (x, y, z))
            vertex_stabilizers.append(operator.to_bsf())

        return np.array(vertex_stabilizers, dtype=np.uint)

    def get_face_X_stabilizers(self) -> np.ndarray:
        face_stabilizers = []

        for (axis, x, y, z) in self.face_index.keys():
            operator = Toric3DPauli(self)
            operator.face('X', axis, (x, y, z))
            face_stabilizers.append(operator.to_bsf())

        return np.array(face_stabilizers, dtype=np.uint)

    def measure_syndrome(self, error: Toric3DPauli) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizers, error.to_bsf())