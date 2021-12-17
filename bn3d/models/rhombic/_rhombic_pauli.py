from typing import Tuple, Optional
import numpy as np
from ..generic._indexed_code import IndexedCodePauli
from qecsim.model import StabilizerCode


class RhombicPauli(IndexedCodePauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        if bsf is not None:
            bsf = bsf.copy()

        super().__init__(code, bsf=bsf)

    def vertex(
        self, operator: str, location: Tuple[int, int, int, int]
    ):
        r"""Apply triangle operator on sites neighbouring vertex (3-body terms).

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int]
            The (axis, x, y, z) location and orientation of the triangle
        """

        axis, x, y, z = location
        
        if not self.code.is_vertex(location):
            raise ValueError(f'Location {location} does not correspond to a triangle')

        # Vertices of type 1 have neighboring cube stabs on the top left and bottom right
        # vertex_1[axis] contains the 3 qubits (i.e. 3 locations) in the corresponding triangle
        vertex_1 = [[(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)],
                    [(x - 1, y, z), (x, y - 1, z), (x, y, z + 1)],
                    [(x + 1, y, z), (x, y - 1, z), (x, y, z - 1)],
                    [(x - 1, y, z), (x, y + 1, z), (x, y, z - 1)]]

        # Vertices of type 2 have neighboring cube stabs on the top right and bottom left
        vertex_2 = [[(x + 1, y, z), (x, y + 1, z), (x, y, z - 1)],
                    [(x - 1, y, z), (x, y - 1, z), (x, y, z - 1)],
                    [(x + 1, y, z), (x, y - 1, z), (x, y, z + 1)],
                    [(x - 1, y, z), (x, y + 1, z), (x, y, z + 1)]]

        vertex = vertex_1 if (x + y + z) % 4 == 0 else vertex_2

        for qubit_location in vertex[axis]:
            mod_location = tuple(np.mod(qubit_location, 2*self.code.size))
            self.site(operator, mod_location)

    def face(
        self, operator: str,
        location: Tuple[int, int, int]
    ):
        r"""Apply cube operator on sites around cubes.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int]
            The (x, y, z) location of the cube
        """

        x, y, z = location
        print("Face", x, y, z)

        if not self.code.is_face(location):
            raise ValueError(f'Location {location} does not correspond to a cube')

        cube = [(x + 1, y + 1, z), (x - 1, y - 1, z), (x + 1, y - 1, z), (x - 1, y + 1, z),
                (x - 1, y, z - 1), (x + 1, y, z - 1), (x, y - 1, z - 1), (x, y + 1, z - 1),
                (x - 1, y, z + 1), (x + 1, y, z + 1), (x, y - 1, z + 1), (x, y + 1, z + 1)]

        for qubit_location in cube:
            mod_location = tuple(np.mod(qubit_location, 2*self.code.size))
            print("Qubit", mod_location)
            self.site(operator, mod_location)