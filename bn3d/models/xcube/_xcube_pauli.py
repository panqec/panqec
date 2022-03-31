from typing import Tuple, Optional
import numpy as np
from ..generic._indexed_sparse_pauli import IndexedSparsePauli
from qecsim.model import StabilizerCode


class XCubePauli(IndexedSparsePauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        if bsf is not None:
            bsf = bsf.copy()

        super().__init__(code, bsf=bsf)

    def vertex(
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

        if not self.code.is_vertex(location):
            raise ValueError(f'Location {location} does not correspond to a cube')

        cube = [(x + 1, y + 1, z), (x - 1, y - 1, z), (x + 1, y - 1, z), (x - 1, y + 1, z),
                (x - 1, y, z - 1), (x + 1, y, z - 1), (x, y - 1, z - 1), (x, y + 1, z - 1),
                (x - 1, y, z + 1), (x + 1, y, z + 1), (x, y - 1, z + 1), (x, y + 1, z + 1)]

        for qubit_location in cube:
            mod_location = tuple(np.mod(qubit_location, 2*np.array(self.code.size)))
            self.site(operator, mod_location)

    def face(
        self, operator: str, location: Tuple[int, int, int, int]
    ):
        r"""Apply face operator on sites neighbouring vertex.

        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int]
            The (axis, x, y, z) location and orientation of the face
        """

        axis, x, y, z = location

        if not self.code.is_face(location):
            raise ValueError(f'Location {location} does not correspond to a face')

        face = [[], [], []]

        face[self.code.X_AXIS] = [(x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]
        face[self.code.Y_AXIS] = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
        face[self.code.Z_AXIS] = [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z)]

        for qubit_location in face[axis]:
            mod_location = tuple(np.mod(qubit_location, 2*np.array(self.code.size)))
            self.site(operator, mod_location)
