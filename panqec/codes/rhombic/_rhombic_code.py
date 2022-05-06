import itertools
from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class RhombicCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Rhombic {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self):
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
            if (x + y + z) % 4 == 1:
                coordinates.append((x, y, z))

        # Triangles
        ranges = [range(4), range(0, 2*Lx, 2), range(0, 2*Ly, 2), range(0, 2*Lz, 2)]
        for axis, x, y, z in itertools.product(*ranges):
            coordinates.append((axis, x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if len(location) == 4:
            return 'triangle'
        else:
            return 'cube'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if self.stabilizer_type(location) == 'cube':
            pauli = 'X'
        else:
            pauli = 'Z'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        if self.stabilizer_type(location) == 'cube':
            x, y, z = location
            delta = [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                     (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
                     (0, 1, 1), (0, -1, -1), (0, -1, 1), (0, 1, -1)]
        else:
            axis, x, y, z = location
            if (x + y + z) % 4 == 0:
                delta_axis = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                              [(-1, 0, 0), (0, -1, 0), (0, 0, 1)],
                              [(1, 0, 0), (0, -1, 0), (0, 0, -1)],
                              [(-1, 0, 0), (0, 1, 0), (0, 0, -1)]]
            else:
                delta_axis = [[(1, 0, 0), (0, 1, 0), (0, 0, -1)],
                              [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
                              [(1, 0, 0), (0, -1, 0), (0, 0, 1)],
                              [(-1, 0, 0), (0, 1, 0), (0, 0, 1)]]
            delta = delta_axis[axis]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add([x, y, z], d) % (2*np.array(self.size)))

            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        if (z % 2 == 0) and (x % 2 == 1) and (y % 2 == 0):
            axis = 'x'
        elif (z % 2 == 0) and (x % 2 == 0) and (y % 2 == 1):
            axis = 'y'
        elif (z % 2 == 1) and (x % 2 == 0) and (y % 2 == 0):
            axis = 'z'
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def get_logicals_x(self) -> Operator:
        """The 3 logical X operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Sheet of X operators normal to the z direction
        operator = dict()
        for x in range(2*Lx):
            for y in range(2*Ly):
                if (x + y) % 2 == 1:
                    operator[(x, y, 0)] = 'X'
        logicals.append(operator)

        # Sheet of X operators normal to the y direction
        operator = dict()
        for x in range(2*Lx):
            for z in range(2*Lz):
                if (x + z) % 2 == 1:
                    operator[(x, 0, z)] = 'X'
        logicals.append(operator)

        # Sheet of X operators normal to the x direction
        operator = dict()
        for y in range(2*Ly):
            for z in range(2*Lz):
                if (y + z) % 2 == 1:
                    operator[(0, y, z)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        """The 3 logical Z operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Line of parallel Z operators along the x direction
        operator = dict()
        for x in range(0, 2*Lx, 2):
            operator[(x, 1, 0)] = 'Z'
        logicals.append(operator)

        # Line of parallel Z operators along the y direction
        operator = dict()
        for y in range(0, 2*Ly, 2):
            operator[(1, y, 0)] = 'Z'
        logicals.append(operator)

        # Line of parallel Z operators along the z direction
        operator = dict()
        for z in range(0, 2*Lz, 2):
            operator[(0, 1, z)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().stabilizer_representation(location, rotated_picture)

        if self.stabilizer_type(location) == 'triangle':
            axis, x, y, z = location
            representation['location'] = [x, y, z]

            delta_1 = [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]
            delta_2 = [[1, 1, -1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1]]

            delta = delta_1 if ((x + y + z) % 4 == 0) else delta_2

            a = 0.5
            dx, dy, dz = tuple(a * np.array(delta[axis]))

            representation['params']['vertices'] = [[dx, 0, 0],
                                                    [0, dy, 0],
                                                    [0, 0, dz]]

        return representation
