from typing import Tuple, Optional
import numpy as np
from qecsim.models.toric import ToricPauli
from qecsim.model import StabilizerCode


class Toric3DPauli(ToricPauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        if bsf is not None:
            bsf = bsf.copy()

        super().__init__(code, bsf=bsf)

    def vertex(self, operator: str, location: Tuple[int, int, int]):
        r"""Apply operator on sites neighbouring vertex.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int]
            The (x, y, z) location of the vertex.

        Examples
        --------
        operator.vertex('X', (0, 0, 0))

             .       .            Coordinate axes:
              \     /                        y
               X   X                        /
                \ /                        /
         .---X---o---X---.         z <----o
                / \                        \
               X   X                        \
              /     \                        x
             .       .
        """

        # Location modulo lattice shape, to handle edge cases.
        x, y, z = np.mod(location, self.code.size)

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

    def face(
        self, operator: str,
        normal: int,
        location: Tuple[int, int, int]
    ):
        r"""Apply operator on sites on face normal to direction at location.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        normal: int
            Direction normal to face.
            0 is x-normal face in yz plane,
            1 is y-normal face in xz plane,
            2 is z-normal face in xy plane.
        location: Tuple[int, int, int]
            The (x, y, z) location of the vertex of the face
            that is closest to the origin.

        Examples
        --------
        operator.face('X', 0, (0, 0, 0))

             .---X---.            Coordinate axes:
            /       /                        y
           X       X   \                    /
          /       /                        /
         .---X---o       .         z <----o
                                           \
           \       \   /                    \
                                             x
             .   -   .
        """

        # Location modulo lattice shape, to handle edge cases.
        x, y, z = np.mod(location, self.code.size)

        X_AXIS, Y_AXIS, Z_AXIS = (
            self.code.X_AXIS, self.code.Y_AXIS, self.code.Z_AXIS
        )

        # x-normal so face is in yz-plane.
        if normal == X_AXIS:
            self.site(operator, (Y_AXIS, x, y, z))
            self.site(operator, (Z_AXIS, x, y + 1, z))
            self.site(operator, (Y_AXIS, x, y, z + 1))
            self.site(operator, (Z_AXIS, x, y, z))

        # y-normal so face is in zx-plane.
        elif normal == Y_AXIS:
            self.site(operator, (Z_AXIS, x, y, z))
            self.site(operator, (X_AXIS, x, y, z + 1))
            self.site(operator, (Z_AXIS, x + 1, y, z))
            self.site(operator, (X_AXIS, x, y, z))

        # z-normal so face is xy-plane.
        elif normal == Z_AXIS:
            self.site(operator, (X_AXIS, x, y, z))
            self.site(operator, (Y_AXIS, x + 1, y, z))
            self.site(operator, (X_AXIS, x, y + 1, z))
            self.site(operator, (Y_AXIS, x, y, z))
