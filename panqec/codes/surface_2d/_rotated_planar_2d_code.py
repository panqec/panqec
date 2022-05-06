from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple[int, int], str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple[int, int]]  # List of locations


class RotatedPlanar2DCode(StabilizerCode):
    dimension = 2

    @property
    def label(self) -> str:
        return 'Rotated Planar {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Operator:
        coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly+1, 2):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly = self.size

        # Vertices
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly+1, 2):
                if (x + y) % 4 == 2:
                    coordinates.append((x, y))

        # Faces
        for x in range(0, 2*Lx+1, 2):
            for y in range(2, 2*Ly, 2):
                if (x + y) % 4 == 0:
                    coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int]):
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y = location
        if (x + y) % 4 == 2:
            return 'vertex'
        else:
            return 'face'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if self.stabilizer_type(location) == 'vertex':
            pauli = 'Z'
        else:
            pauli = 'X'

        deformed_pauli = {'Z': 'X', 'X': 'Z'}[pauli]

        delta = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location):
        x, y = location

        if (x + y) % 4 == 2:
            axis = 'x'
        elif (x + y) % 4 == 0:
            axis = 'y'
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def get_logicals_x(self) -> np.ndarray:
        Lx, Ly = self.size
        logicals = []

        # X operators along first diagonal
        operator = dict()
        for x in range(1, 2*Lx+1, 2):
            operator[(x, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> np.ndarray:
        Lx, Ly = self.size
        logicals = []

        # Z operators along first diagonal
        operator = dict()
        for y in range(1, 2*Ly+1, 2):
            operator[(1, y)] = 'Z'
        logicals.append(operator)

        return logicals
