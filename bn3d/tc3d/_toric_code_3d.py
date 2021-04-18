import itertools
from typing import Tuple, Optional
import numpy as np
from qecsim.model import StabilizerCode
from ._toric_3d_pauli import Toric3DPauli


class ToricCode3D(StabilizerCode):

    _shape: Tuple[int, int, int, int]
    X_AXIS: int = 0
    Y_AXIS: int = 1
    Z_AXIS: int = 2
    _stabilizers = np.array([])
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

    # StabilizerCode interface methods.

    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        return (np.product(self.shape), 3, min(self.size))

    @property
    def label(self) -> str:
        return ''

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

    # TODO: ToricCode3D specific methods.
    def get_vertex_Z_stabilizers(self) -> np.ndarray:
        vertex_stabilizers = []
        ranges = [range(length) for length in self.size]

        # Z operators for each vertex for each position.
        for L_x, L_y, L_z in itertools.product(*ranges):
            operator = Toric3DPauli(self)
            operator.vertex('Z', (L_x, L_y, L_z))
            vertex_stabilizers.append(operator.to_bsf())
        return np.array(vertex_stabilizers, dtype=np.uint)

    def get_face_X_stabilizers(self) -> np.ndarray:
        face_stabilizers = []
        ranges = [range(length) for length in self.shape]

        # X operators for each normal direction and for each face position.
        for normal, L_x, L_y, L_z in itertools.product(*ranges):
            operator = Toric3DPauli(self)
            operator.face('X', normal, (L_x, L_y, L_z))
            face_stabilizers.append(operator.to_bsf())
        return np.array(face_stabilizers, dtype=np.uint)
