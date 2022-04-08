from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode
import numpy as np

Operator = Dict[Tuple[int, int], str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple[int, int]]  # List of locations


class Planar2DCode(StabilizerCode):
    dimension = 2

    @property
    def label(self) -> str:
        return 'Planar {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
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

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly = self.size

        # Vertices
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Faces
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int]) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y = location
        if x % 2 == 0:
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

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location: Tuple[int, int]) -> int:
        x, y = location

        if (x % 2 == 1) and (y % 2 == 0):
            axis = 'x'
        elif (x % 2 == 0) and (y % 2 == 1):
            axis = 'y'
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def get_logicals_x(self) -> Operator:
        Lx, Ly = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        Lx, Ly = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for y in range(0, 2*Ly, 2):
            operator[(1, y)] = 'Z'
        logicals.append(operator)

        return logicals
