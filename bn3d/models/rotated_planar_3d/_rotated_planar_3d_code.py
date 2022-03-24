from typing import Tuple, Dict
from bn3d.models import StabilizerCode
import numpy as np


class RotatedPlanar3DCode(StabilizerCode):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def label(self) -> str:
        return 'Rotated Planar {}x{}x{}'.format(*self.size)

    def _vertex(self, location: Tuple[int, int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        x, y, z = location

        if location not in self.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        pauli = 'Z'
        deformed_pauli = 'X'

        delta = [(-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def _face(self, location: Tuple[int, int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        x, y, z = location

        if location not in self.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        pauli = 'X'
        deformed_pauli = 'Z'

        # z-normal so face is xy-plane.
        if z % 2 == 1:
            delta = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]
        # x-normal so face is in yz-plane.
        elif (x + y) % 4 == 0:
            delta = [(-1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
        # y-normal so face is in zx-plane.
        elif (x + y) % 4 == 2:
            delta = [(-1, 1, 0), (1, -1, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def axis(self, location):
        x, y, z = location

        if location not in self.qubit_index.keys():
            raise ValueError(f'Location {location} does not correspond to a qubit')

        if (z % 2 == 0):
            axis = self.Z_AXIS
        elif (x + y) % 4 == 2:
            axis = self.X_AXIS
        elif (x + y) % 4 == 0:
            axis = self.Y_AXIS

        return axis

    def _get_qubit_coordinates(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(2, 2*Lz, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_vertex_coordinates(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        for z in range(1, 2*Lz, 2):
            for x in range(2, 2*Lx, 2):
                for y in range(0, 2*Ly+1, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_face_coordinates(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal faces
        for x in range(0, 2*Lx+1, 2):
            for y in range(2, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))
        # Vertical faces
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly+1, 2):
                for z in range(2, 2*Lz, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_logicals_x(self) -> Dict[str, Tuple]:
        """Get the unique logical X operator."""
        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, min(2*Lx, 2*Ly), 2):
            operator[(x, 2*Ly - x, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def _get_logicals_z(self) -> Dict[str, Tuple]:
        """Get the unique logical Z operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for z in range(1, 2*Lz, 2):
            for x in range(1, min(2*Lx, 2*Ly), 2):
                operator[(x, x, z)] = 'Z'
        logicals.append(operator)

        return logicals


if __name__ == "__main__":
    code = RotatedPlanar3DCode(2)

    print("Vertices", code.face_index)
