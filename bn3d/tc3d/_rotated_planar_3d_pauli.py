from typing import Tuple, Optional
import numpy as np
from qecsim.model import StabilizerCode


class RotatedPlanar3DPauli:
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        self._code = code
        self._from_bsf(bsf)

    def _from_bsf(self, bsf):
        # initialise lattices for X and Z operators from bsf
        n_qubits = self.code.n_k_d[0]
        if bsf is None:
            # initialise identity lattices for X and Z operators
            self._xs = np.zeros(n_qubits, dtype=int)
            self._zs = np.zeros(n_qubits, dtype=int)
        else:
            assert len(bsf) == 2 * n_qubits, 'BSF {} has incompatible length'.format(bsf)
            assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)
            # initialise lattices for X and Z operators from bsf
            self._xs, self._zs = np.hsplit(bsf, 2)  # split out Xs and Zs

    def get_index(self, coordinate):
        if coordinate not in self.code.qubit_index.keys():
            raise ValueError(f"Incorrect qubit coordinate {coordinate} given when constructing the operator")
        return self.code.qubit_index[coordinate]

    def site(self, operator, *indices):
        """
        Apply the operator to site identified by the index.
        Notes:
        * Index is in the format (x, y).
        * Index is modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1) indexes the same site as (0, 1).
        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param indices: Any number of indices identifying sites in the format (x, y).
        :type indices: Any number of 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanar3DPauli
        """
        for coord in indices:
            # flip sites
            flat_index = self.get_index(coord)
            if operator in ('X', 'Y'):
                self._xs[flat_index] ^= 1
            if operator in ('Z', 'Y'):
                self._zs[flat_index] ^= 1
        return self

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
            if x + 1 <= 4*Lx+1 and y + 1 <= 4*Ly+2:
                self.site(operator, (x + 1, y + 1, z))
            if x - 1 > 0 and y + 1 <= 4*Ly+2:
                self.site(operator, (x - 1, y + 1, z))
            if x + 1 <= 4*Lx+1 and y - 1 >= 0:
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

    @property
    def code(self):
        """
        The rotated toric code.
        :rtype: RotatedToricCode
        """
        return self._code

    def operator(self, coord):
        """
        Returns the operator on the site identified by the coordinates.
        Notes:
        * coord is in the format (x, y, z).
        :param coord: Coordinate identifying a site in the format (x, y, z).
        :type  coord: 3-tuple of int
        :return: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :rtype: str
        """
        # extract binary x and z
        index = self.code.qubit_index[coord]
        x = self._xs[index]
        z = self._zs[index]
        # return Pauli
        if x == 1 and z == 1:
            return 'Y'
        if x == 1:
            return 'X'
        if z == 1:
            return 'Z'
        else:
            return 'I'

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.
        :return: A copy of this Pauli.
        :rtype: ToricPauli
        """
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def __eq__(self, other):
        if type(other) is type(self):
            return np.array_equal(self._xs, other._xs) and np.array_equal(self._zs, other._zs)
        return NotImplemented

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.code, self.to_bsf())

    def __str__(self):
        """
        ASCII art style lattice showing primal lattice lines and Pauli operators.
        :return: Informal string representation.
        :rtype: str
        """
        return self.code.ascii_art(pauli=self)

    def to_bsf(self):
        """
        Binary symplectic representation of Pauli.
        Notes:
        * For performance reasons, the returned bsf is a view of this Pauli. Modifying one will modify the other.
        :return: Binary symplectic representation of Pauli.
        :rtype: numpy.array (1d)
        """
        return np.concatenate((self._xs.flatten(), self._zs.flatten()))