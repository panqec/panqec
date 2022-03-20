from typing import Tuple
from ..generic._indexed_sparse_pauli import IndexedSparsePauli
import numpy as np


class Toric2DPauli(IndexedSparsePauli):
    """Pauli Operator on 2D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def vertex(self, operator: str, location: Tuple[int, int], deformed_axis=None):
        r"""Apply operator on sites neighboring vertex.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int]
            The (x, y) location of the vertex.

        Examples
        --------
        operator.vertex('X', (0, 0))

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

        Lx, Ly = self.code.size
        x, y = location

        if not self.code.is_vertex(location):
            raise ValueError(f"Incorrect coordinate {location} for a vertex")

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for d in delta:
            qubit_location = tuple(np.add(location, d) % (2*np.array(self.code.size)))

            is_deformed = (self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)

    def face(self, operator: str, location: Tuple[int, int], deformed_axis=None):
        r"""Apply operator on sites on face normal to direction at location.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int])
            The (x, y) location of the vertex of the face
            that is closest to the origin.

        Examples
        --------
        operator.face('X', (1, 1))

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

        if not self.code.is_face(location):
            raise ValueError(f"Incorrect coordinate {location} for a face")

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for d in delta:
            qubit_location = tuple(np.add(location, d) % (2*np.array(self.code.size)))
            is_deformed = (self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)
