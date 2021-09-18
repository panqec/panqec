from typing import Tuple, Optional
import numpy as np
from qecsim.models.toric import ToricPauli
from qecsim.model import StabilizerCode


class RhombicPauli(ToricPauli):
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

    def triangle(self, operator: str, axis: int, location: Tuple[int, int, int]):
        r"""Apply operator on sites neighbouring vertex (3-body terms).

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
        if ((x + y + z) % 2 == 0):
            if axis == 0:
                self.site(operator, (0, x, y, z))
                self.site(operator, (1, x, y, z))
                self.site(operator, (2, x, y, z))
            elif axis == 1:
                self.site(operator, (0, x, y, z))
                self.site(operator, (1, x, y-1, z))
                self.site(operator, (2, x, y, z-1))
            elif axis == 2:
                self.site(operator, (0, x-1, y, z))
                self.site(operator, (1, x, y, z))
                self.site(operator, (2, x, y, z-1))
            elif axis == 3:
                self.site(operator, (0, x-1, y, z))
                self.site(operator, (1, x, y-1, z))
                self.site(operator, (2, x, y, z))
                
        else:
            if axis == 0:
                self.site(operator, (0, x-1, y, z))
                self.site(operator, (1, x, y-1, z))
                self.site(operator, (2, x, y, z-1))
            elif axis == 1:
                self.site(operator, (0, x-1, y, z))
                self.site(operator, (1, x, y, z))
                self.site(operator, (2, x, y, z))
            elif axis == 2:
                self.site(operator, (0, x, y, z))
                self.site(operator, (1, x, y-1, z))
                self.site(operator, (2, x, y, z))
            elif axis == 3:
                self.site(operator, (0, x, y, z))
                self.site(operator, (1, x, y, z))
                self.site(operator, (2, x, y, z-1))
        
    def cube(
        self, operator: str,
        location: Tuple[int, int, int]
    ):
        r"""Apply operator on sites around cubes.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int]
            The (x, y, z) location of the vertex of the face
            that is closest to the origin.

        Examples
        --------
        operator.cube('X', 0, (0, 0, 0))

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

        self.site(operator, (X_AXIS, x, y, z))
        self.site(operator, (X_AXIS, x, y, z+1))
        self.site(operator, (X_AXIS, x, y+1, z))
        self.site(operator, (X_AXIS, x, y+1, z+1))
        
        self.site(operator, (Y_AXIS, x, y, z))
        self.site(operator, (Y_AXIS, x+1, y, z))
        self.site(operator, (Y_AXIS, x, y, z+1))
        self.site(operator, (Y_AXIS, x+1, y, z+1))
        
        self.site(operator, (Z_AXIS, x, y, z))
        self.site(operator, (Z_AXIS, x+1, y, z))
        self.site(operator, (Z_AXIS, x, y+1, z))
        self.site(operator, (Z_AXIS, x+1, y+1, z))

