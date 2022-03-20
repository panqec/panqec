from typing import Tuple, Optional
import numpy as np
from ..generic._indexed_sparse_pauli import IndexedSparsePauli
from qecsim.model import StabilizerCode


class RhombicPauli(IndexedSparsePauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        if bsf is not None:
            bsf = bsf.copy()

        super().__init__(code, bsf=bsf)

    def vertex(self, operator: str, location: Tuple[int, int, int, int], deformed_axis=None):
        r"""Apply triangle operator on sites neighboring vertex (3-body terms).

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
        delta_1 = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                   [(-1, 0, 0), (0, -1, 0), (0, 0, 1)],
                   [(1, 0, 0), (0, -1, 0), (0, 0, -1)],
                   [(-1, 0, 0), (0, 1, 0), (0, 0, -1)]]

        # Vertices of type 2 have neighboring cube stabs on the top right and bottom left
        delta_2 = [[(1, 0, 0), (0, 1, 0), (0, 0, -1)],
                   [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
                   [(1, 0, 0), (0, -1, 0), (0, 0, 1)],
                   [(-1, 0, 0), (0, 1, 0), (0, 0, 1)]]

        delta = delta_1 if (x + y + z) % 4 == 0 else delta_2

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        for d in delta[axis]:
            qubit_location = tuple(np.add([x, y, z], d) % (2*np.array(self.code.size)))
            qx, qy, qz = qubit_location

            is_deformed = ((qx + qy + qz) % 4 == 1 and self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)

    def face(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
        r"""Apply cube operator on sites around cubes.
        Parameters
        ----------
        operator: str
            Pauli operator in string format.
        location: Tuple[int, int, int]
            The (x, y, z) location of the cube
        """

        if not self.code.is_face(location):
            raise ValueError(f'Location {location} does not correspond to a cube')

        delta = [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                 (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
                 (0, 1, 1), (0, -1, -1), (0, -1, 1), (0, 1, -1)]

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        for d in delta:
            qubit_location = tuple(np.add(location, d) % (2*np.array(self.code.size)))
            qx, qy, qz = qubit_location

            is_deformed = ((qx + qy + qz) % 4 == 1 and self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)
