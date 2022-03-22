from typing import Tuple, Optional
import numpy as np
from bn3d.models import StabilizerPauli
from qecsim.model import StabilizerCode


class XCubePauli(StabilizerPauli):
    """Pauli Operator on 3D Toric Code.

    Qubit sites are on edges of the lattice.
    """

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        if bsf is not None:
            bsf = bsf.copy()

        super().__init__(code, bsf=bsf)

    def vertex(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
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

        delta = [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                 (-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1),
                 (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        for d in delta:
            qubit_location = tuple(np.add(location, d) % (2*np.array(self.code.size)))

            is_deformed = (self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)

    def face(self, operator: str, location: Tuple[int, int, int, int], deformed_axis=None):
        r"""Apply face operator on sites neighboring vertex.

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

        delta = [[], [], []]

        delta[self.code.X_AXIS] = [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        delta[self.code.Y_AXIS] = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
        delta[self.code.Z_AXIS] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        for d in delta[axis]:
            qubit_location = tuple(np.add([x, y, z], d) % (2*np.array(self.code.size)))

            is_deformed = (self.code.axis(qubit_location) == deformed_axis)

            if is_deformed:
                self.site(deformed_operator, qubit_location)
            else:
                self.site(operator, qubit_location)
