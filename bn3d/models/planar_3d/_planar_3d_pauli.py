from typing import Tuple, Optional
import numpy as np
from ..generic._indexed_sparse_pauli import IndexedSparsePauli
from qecsim.model import StabilizerCode


class Planar3DPauli(IndexedSparsePauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

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

        x, y, z = location

        if (x + y + z) % 2 == 1:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        vertex = [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]

        for qubit_location in vertex:
            if qubit_location in self.code.qubit_index:
                self.site(operator, qubit_location)

    def face(
        self, operator: str,
        location: Tuple[int, int, int]
    ):
        r"""Apply operator on sites on face normal to direction at location.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int])
            The (x, y, z) location of the vertex of the face
            that is closest to the origin.

        Examples
        --------
        operator.face('X', (0, 0, 0))

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

        x, y, z = location

        z_face = [(x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z)]  # zy-plane
        x_face = [(x, y - 1, z), (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)]  # xz-plane
        y_face = [(x - 1, y, z), (x + 1, y, z), (x, y, z - 1), (x, y, z + 1)]  # xy-plane

        # Choose the right face orientation depending on its position
        if z % 2 == 0:
            face = z_face
        elif (x % 2 == 0):
            face = x_face
        elif (y % 2 == 0):
            face = y_face
        else:
            raise ValueError(f"Invalid coordinate {location} for a face")

        # Place the qubits
        for index in face:
            if index in self.code.qubit_index:
                self.site(operator, index)
