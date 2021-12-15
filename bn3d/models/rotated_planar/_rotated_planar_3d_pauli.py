from typing import Tuple
from ..generic._indexed_code import IndexedCodePauli


class RotatedPlanar3DPauli(IndexedCodePauli):
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

        x, y, z = location
        Lx, Ly, Lz = self.code.size

        if (x + y) % 4 != 2 or z % 2 == 0:
            raise ValueError(f"Incorrect coordinate {location} for a vertex")

        # Four horizontal edges (at most)
        if x - 1 >= 0 and y - 1 >= 0:
            self.site(operator, (x - 1, y - 1, z))
        if x + 1 <= 4*Lx+1 and y + 1 <= 4*Ly+2:
            self.site(operator, (x + 1, y + 1, z))
        if x - 1 > 0 and y + 1 <= 4*Ly+2:
            self.site(operator, (x - 1, y + 1, z))
        if x + 1 <= 4*Lx+1 and y - 1 >= 0:
            self.site(operator, (x + 1, y - 1, z))

        # Two vertical edges
        if z - 1 >= 1:
            self.site(operator, (x, y, z - 1))
        if z + 1 < 2*Lz+1:
            self.site(operator, (x, y, z + 1))

    def face(self, operator: str, location: Tuple[int, int, int]):
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
        x, y, z = location
        Lx, Ly, Lz = self.code.size

        # Horizontal face
        if z % 2 == 1:
            if x - 1 >= 0 and y - 1 >= 0:
                self.site(operator, (x - 1, y - 1, z))
            if x + 1 <= 4*Lx+2 and y + 1 <= 4*Ly+2:
                self.site(operator, (x + 1, y + 1, z))
            if x - 1 > 0 and y + 1 <= 4*Ly+2:
                self.site(operator, (x - 1, y + 1, z))
            if x + 1 <= 4*Lx+2 and y - 1 >= 0:
                self.site(operator, (x + 1, y - 1, z))

        # Vertical face (axis /)
        elif (x + y) % 4 == 0:
            self.site(operator, (x, y, z - 1))
            self.site(operator, (x, y, z + 1))
            if x - 1 > 0 and y - 1 >= 0:
                self.site(operator, (x - 1, y - 1, z))
            if x + 1 <= 4*Lx+1 and y + 1 <= 4*Ly+2:
                self.site(operator, (x + 1, y + 1, z))

        # Vertical face (axis \)
        elif (x + y) % 4 == 2:
            self.site(operator, (x, y, z - 1))
            self.site(operator, (x, y, z + 1))
            if x - 1 > 0 and y + 1 <= 4*Ly+2:
                self.site(operator, (x - 1, y + 1, z))
            if x + 1 <= 4*Lx+1 and y - 1 >= 0:
                self.site(operator, (x + 1, y - 1, z))

        else:
            raise ValueError(f"Invalid coordinate {location} for a face")
