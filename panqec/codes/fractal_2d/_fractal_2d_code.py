from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple[int, int], str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple[int, int]]  # List of locations


class Fractal2DCode(StabilizerCode):
    dimension = 2

    @property
    def label(self) -> str:
        return 'Fractal {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly = self.size

        for x in range(0, Lx, 1):
            for y in range(0, Ly, 1):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = self.qubit_coordinates.copy()

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int]) -> str:
        return 'tetris'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        pauli = 'Z'

        delta = [(-1, 0), (0, 0), (1, 0), (0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))
            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location) -> int:
        return 0

    def get_logicals_x(self) -> Operator:
        """The 2 logical X operators."""

        Lx, Ly = self.size
        logicals = []

        operator = dict()
        operator[(0, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        """The 2 logical Z operators."""

        Lx, Ly = self.size
        logicals = []

        operator = dict()
        operator[(0, 0)] = 'Z'
        logicals.append(operator)

        return logicals
