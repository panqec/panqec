from typing import Tuple, Dict
from bn3d.models import StabilizerCode
import numpy as np

Indexer = Dict[Tuple[int, int], int]  # coordinate to index


class Planar2DCode(StabilizerCode):
    @property
    def dimension(self) -> int:
        return 2

    @property
    def label(self) -> str:
        return 'Toric {}x{}'.format(*self.size)

    def _vertex(self, location: Tuple[int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        x, y = location

        if (x, y) not in self.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        pauli = 'Z'
        deformed_pauli = 'X'

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def _face(self, location: Tuple[int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        x, y = location

        if (x, y) not in self.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        pauli = 'X'
        deformed_pauli = 'Z'

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def axis(self, location: Tuple[int, int]) -> int:
        x, y = location

        if (x % 2 == 1) and (y % 2 == 0):
            axis = self.X_AXIS
        elif (x % 2 == 0) and (y % 2 == 1):
            axis = self.Y_AXIS
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def _get_qubit_coordinates(self) -> Indexer:
        coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Qubits along e_y
        for x in range(2, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_vertex_coordinates(self) -> Indexer:
        coordinates = []
        Lx, Ly = self.size

        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_face_coordinates(self) -> Indexer:
        coordinates = []
        Lx, Ly = self.size

        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_logicals_x(self) -> Dict[str, Tuple]:
        Lx, Ly = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def _get_logicals_z(self) -> Dict[str, Tuple]:
        Lx, Ly = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for y in range(0, 2*Ly, 2):
            operator[(1, y)] = 'Z'
        logicals.append(operator)

        return logicals
