from typing import Tuple
import numpy as np
from qecsim.models.toric import ToricPauli
from ._toric_code_3d import ToricCode3D


class Toric3DPauli(ToricPauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def __init__(self, code: ToricCode3D, bsf: np.ndarray = None):
        super().__init__(code, bsf)

    def vertex(self, operator: str, location: Tuple[int, int, int]):
        """Apply operator on sites neighbouring vertex."""

        # Location modulo lattice shape, to handle edge cases.
        x, y, z = np.mod(location, self.code.size)
        L_x, L_y, L_z = self.code.size

        # Apply operator on each of the six neighbouring edges.

        # x-edge.
        self.site(operator, (0, x, y, z))
        self.site(operator, (0, x - 1, y, z))

        # y-edge.
        self.site(operator, (1, x, y, z))
        self.site(operator, (1, x, y - 1, z))

        # z-edge.
        self.site(operator, (2, x, y, z))
        self.site(operator, (2, x, y, z - 1))

    def face(self, operator: str, location: Tuple[int, int, int, int]):
        """Apply operator on sites neighbouring face."""
        pass
