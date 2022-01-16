from typing import Tuple
from ..generic._indexed_sparse_pauli import IndexedSparsePauli


class Toric2DPauli(IndexedSparsePauli):
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

        Lx, Ly = self.code.size
        x, y = location

        if (x, y) not in self.code.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        # Apply operator on each of the six neighbouring edges.

        # x-edge.
        self.site(operator, ((x + 1) % (2*Lx), y))
        self.site(operator, ((x - 1) % (2*Lx), y))

        # y-edge.
        self.site(operator, (x, (y + 1) % (2*Ly)))
        self.site(operator, (x, (y - 1) % (2*Ly)))

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
        Lx, Ly = self.code.size

        if (x, y) not in self.code.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        self.site(operator, ((x - 1) % (2*Lx), y))
        self.site(operator, ((x + 1) % (2*Lx), y))
        self.site(operator, (x, (y - 1) % (2*Ly)))
        self.site(operator, (x, (y + 1) % (2*Ly)))
