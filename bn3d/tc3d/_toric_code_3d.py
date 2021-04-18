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
        return np.array([], dtype=np.uint)

    @property
    def logical_xs(self) -> np.ndarray:
        return np.array([], dtype=np.uint)

    @property
    def logical_zs(self) -> np.ndarray:
        return np.array([], dtype=np.uint)

    @property
    def size(self) -> Tuple[int, int, int]:
        """Dimensions of lattice."""
        return self._shape[1:]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Shape of lattice for each qubit."""
        return self._shape

    # TODO: ToricCode3D specific methods.
    def get_vertex_Z_stabilizers(self):
        vertex_stabilizers = []
        ranges = [range(length) for length in self.size]
        for position in itertools.product(*ranges):
            operator = Toric3DPauli(self)
            operator.vertex('Z', position)
            vertex_stabilizers.append(operator.to_bsf())
        return np.array(vertex_stabilizers, dtype=np.uint)
