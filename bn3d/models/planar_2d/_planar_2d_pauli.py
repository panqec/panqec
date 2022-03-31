from typing import Tuple
from ..generic._indexed_sparse_pauli import IndexedSparsePauli


class Planar2DPauli(IndexedSparsePauli):
    """Pauli Operator on 2D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def vertex(self, operator: str, location: Tuple[int, int]):
        r"""Apply operator on sites neighbouring vertex.

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

        x, y = location

        if (x, y) not in self.code.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(len(delta)):
            location = (x + delta[i][0], y + delta[i][1])
            if location in self.code.qubit_index:
                self.site(operator, location)

    def face(
        self, operator: str,
        location: Tuple[int, int]
    ):
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

        # Location modulo lattice shape, to handle edge cases.
        x, y = location

        if (x, y) not in self.code.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(len(delta)):
            location = (x + delta[i][0], y + delta[i][1])
            if location in self.code.qubit_index:
                self.site(operator, location)
