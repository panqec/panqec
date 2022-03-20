from typing import Tuple
from ..generic._indexed_sparse_pauli import IndexedSparsePauli
import numpy as np


class Toric3DPauli(IndexedSparsePauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def vertex(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
        r"""Apply operator on sites neighboring vertex.

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

        if not self.code.is_vertex(location):
            raise ValueError(f"Incorrect coordinate {location} for a vertex")

        delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        for d in delta:
            qubit_location = tuple(np.add(location, d) % (2*np.array(self.code.size)))

            is_deformed = (self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)

    def face(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
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

        # Location modulo lattice shape, to handle edge cases.
        x, y, z = location
        Lx, Ly, Lz = self.code.size

        if not self.code.is_face(location):
            raise ValueError(f"Incorrect coordinate {location} for a face")

        # z-normal so face is xy-plane.
        if z % 2 == 0:
            delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
        # x-normal so face is in yz-plane.
        elif (x % 2 == 0):
            delta = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        # y-normal so face is in zx-plane.
        elif (y % 2 == 0):
            delta = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        for d in delta:
            qubit_location = tuple(np.add(location, d) % (2*np.array(self.code.size)))
            is_deformed = (self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)
