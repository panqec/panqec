from typing import Tuple
from ..generic._indexed_sparse_pauli import IndexedSparsePauli


class Toric3DPauli(IndexedSparsePauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

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

        Lx, Ly, Lz = self.code.size
        x, y, z = location

        if (x + y + z) % 2 == 1:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        # Apply operator on each of the six neighbouring edges.

        # x-edge.
        self.site(operator, ((x + 1) % (2*Lx), y, z))
        self.site(operator, ((x - 1) % (2*Lx), y, z))

        # y-edge.
        self.site(operator, (x, (y + 1) % (2*Ly), z))
        self.site(operator, (x, (y - 1) % (2*Ly), z))

        # z-edge.
        self.site(operator, (x, y, (z + 1) % (2*Lz)))
        self.site(operator, (x, y, (z - 1) % (2*Lz)))

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

        # Location modulo lattice shape, to handle edge cases.
        x, y, z = location
        Lx, Ly, Lz = self.code.size

        # z-normal so face is xy-plane.
        if z % 2 == 0:
            self.site(operator, ((x - 1) % (2*Lx), y, z))
            self.site(operator, ((x + 1) % (2*Lx), y, z))
            self.site(operator, (x, (y - 1) % (2*Ly), z))
            self.site(operator, (x, (y + 1) % (2*Ly), z))

        # x-normal so face is in yz-plane.
        elif (x % 2 == 0):
            self.site(operator, (x, (y - 1) % (2*Ly), z))
            self.site(operator, (x, (y + 1) % (2*Ly), z))
            self.site(operator, (x, y, (z - 1) % (2*Lz)))
            self.site(operator, (x, y, (z + 1) % (2*Lz)))

        # y-normal so face is in zx-plane.
        elif (y % 2 == 0):
            self.site(operator, (x, y, (z - 1) % (2*Lz)))
            self.site(operator, (x, y, (z + 1) % (2*Lz)))
            self.site(operator, ((x - 1) % (2*Lx), y, z))
            self.site(operator, ((x + 1) % (2*Lx), y, z))
        else:
            raise ValueError(f"Invalid coordinate {location} for a face")
