from typing import Tuple
from bn3d.models import StabilizerPauli
import numpy as np


class Planar2DPauli(StabilizerPauli):
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
        """

        x, y = location

        if (x, y) not in self.code.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.code.is_qubit(qubit_location):
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
        """

        # Location modulo lattice shape, to handle edge cases.
        x, y = location

        if (x, y) not in self.code.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        deformed_map = {'X': 'Z', 'Z': 'X'}
        deformed_operator = deformed_map[operator]

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.code.is_qubit(qubit_location):
                is_deformed = (self.code.axis(qubit_location) == deformed_axis)

                if is_deformed:
                    self.site(deformed_operator, qubit_location)
                else:
                    self.site(operator, qubit_location)
