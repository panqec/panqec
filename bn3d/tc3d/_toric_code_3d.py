from typing import Tuple, Optional
import numpy as np
from qecsim.model import StabilizerCode


class ToricCode3D(StabilizerCode):

    _shape: Tuple[int, int, int, int]

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
        return (0, 0, 0)

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
