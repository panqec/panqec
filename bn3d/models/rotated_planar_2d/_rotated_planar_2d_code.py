from typing import Tuple, Dict
import numpy as np
from bn3d.models import StabilizerCode

Indexer = Dict[Tuple[int, int], int]  # coordinate to index


class RotatedPlanar2DCode(StabilizerCode):
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

        delta = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

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

        delta = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def axis(self, location):
        x, y = location

        if (x + y) % 4 == 2:
            axis = self.X_AXIS
        elif (x + y) % 4 == 0:
            axis = self.Y_AXIS
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def _create_qubit_indices(self) -> Indexer:
        coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly+1, 2):
                coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self) -> Indexer:
        coordinates = []
        Lx, Ly = self.size

        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly+1, 2):
                if (x + y) % 4 == 2:
                    coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self) -> Indexer:
        coordinates = []
        Lx, Ly = self.size

        for x in range(0, 2*Lx+1, 2):
            for y in range(2, 2*Ly, 2):
                if (x + y) % 4 == 0:
                    coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index
    
    def _get_logicals_x(self) -> np.ndarray:
        """The 2 logical X operators."""

        Lx, Ly = self.size
        logicals = []

        # X operators along first diagonal
        operator = dict()
        for x in range(1, 2*Lx+1, 2):
            operator[(x, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def _get_logicals_z(self) -> np.ndarray:
        """Get the 3 logical Z operators."""
        Lx, Ly = self.size
        logicals = []

        # Z operators along first diagonal
        operator = dict()
        for y in range(1, 2*Ly+1, 2):
            operator[(1, y)] = 'Z'
        logicals.append(operator)

        return logicals
