import itertools
from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class XCubeCode(StabilizerCode):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def label(self) -> str:
        return 'XCube {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Cubes
        ranges = [range(1, 2*Lx, 2), range(1, 2*Ly, 2), range(1, 2*Lz, 2)]
        for x, y, z in itertools.product(*ranges):
            coordinates.append((x, y, z))

        # Faces
        ranges = [range(3), range(0, 2*Lx, 2), range(0, 2*Ly, 2), range(0, 2*Lz, 2)]
        for axis, x, y, z in itertools.product(*ranges):
            coordinates.append((axis, x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int, int]) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if len(location) == 4:
            return 'face'
        else:
            return 'cube'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if self.stabilizer_type(location) == 'cube':
            pauli = 'Z'
        else:
            pauli = 'X'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        if self.stabilizer_type(location) == 'cube':
            x, y, z = location
            delta = [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                     (-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1),
                     (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]
        else:
            axis, x, y, z = location
            if axis == self.X_AXIS:
                delta = [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            elif axis == self.Y_AXIS:
                delta = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
            else:
                delta = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add([x, y, z], d) % (2*np.array(self.size)))

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def axis(self, location):
        x, y, z = location

        if (z % 2 == 0) and (x % 2 == 1) and (y % 2 == 0):
            axis = self.X_AXIS
        elif (z % 2 == 0) and (x % 2 == 0) and (y % 2 == 1):
            axis = self.Y_AXIS
        elif (z % 2 == 1) and (x % 2 == 0) and (y % 2 == 0):
            axis = self.Z_AXIS
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def get_logicals_x(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        # String of parallel X operators along the x direction
        operator = dict()
        for x in range(0, 2*Lx, 2):
            operator[(x, 1, 0)] = 'X'
        logicals.append(operator)

        # String of parallel X operators normal to the y direction
        operator = dict()
        for y in range(0, 2*Ly, 2):
            operator[(1, y, 0)] = 'X'
        logicals.append(operator)

        # String of parallel X operators normal to the z direction
        operator = dict()
        for z in range(0, 2*Lz, 2):
            operator[(0, 1, z)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        # Line of parallel Z operators along the x direction
        operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 0, 0)] = 'Z'
        logicals.append(operator)

        # Line of parallel Z operators along the y direction
        operator = dict()
        for y in range(1, 2*Ly, 2):
            operator[(0, y, 0)] = 'Z'
        logicals.append(operator)

        # Line of parallel Z operators along the z direction
        operator = dict()
        for z in range(1, 2*Lz, 2):
            operator[(0, 0, z)] = 'Z'
        logicals.append(operator)

        return logicals
