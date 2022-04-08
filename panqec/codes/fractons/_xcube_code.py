import itertools
from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class XCubeCode(StabilizerCode):
    dimension = 3

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

        Lx, Ly, Lz = self.size
        operator = dict()
        for d in delta:
            qubit_location = ((x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly), (z + d[2]) % (2*Lz))
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
        Lx, Ly, Lz = self.size
        logicals = []

        # Line of parallel Z operators along the x direction
        for y in range(0, 2*Ly, 2):
            operator = dict()
            for z in range(0, 2*Lz, 2):
                operator[(1, y, z)] = 'X'
            logicals.append(operator)

        for z in range(2, 2*Lz, 2):
            operator = dict()
            for y in range(0, 2*Ly, 2):
                operator[(1, y, z)] = 'X'
            logicals.append(operator)

        # Line of parallel Z operators along the y direction
        for x in range(0, 2*Lx, 2):
            operator = dict()
            for z in range(0, 2*Lz, 2):
                operator[(x, 1, z)] = 'X'
            logicals.append(operator)

        for z in range(2, 2*Lz, 2):
            operator = dict()
            for x in range(0, 2*Lx, 2):
                operator[(x, 1, z)] = 'X'
            logicals.append(operator)

        # Line of parallel Z operators along the z direction
        for x in range(0, 2*Lx, 2):
            operator = dict()
            for y in range(0, 2*Ly, 2):
                operator[(x, y, 1)] = 'X'
            logicals.append(operator)

        for y in range(2, 2*Ly, 2):
            operator = dict()
            for x in range(0, 2*Lx, 2):
                operator[(x, y, 1)] = 'X'
            logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        # Line of Z operators along the x direction
        for y in range(0, 2*Ly, 2):
            operator = dict()
            for x in range(1, 2*Lx, 2):
                operator[(x, y, 0)] = 'Z'
            logicals.append(operator)

        for z in range(2, 2*Lz, 2):
            operator = dict()
            for x in range(1, 2*Lx, 2):
                operator[(x, 0, z)] = 'Z'
            logicals.append(operator)

        # Line of Z operators along the y direction
        for x in range(0, 2*Lx, 2):
            operator = dict()
            for y in range(1, 2*Ly, 2):
                operator[(x, y, 0)] = 'Z'
            logicals.append(operator)

        for z in range(2, 2*Lz, 2):
            operator = dict()
            for y in range(1, 2*Ly, 2):
                operator[(0, y, z)] = 'Z'
            logicals.append(operator)

        # Line of Z operators along the z direction
        for x in range(0, 2*Lx, 2):
            operator = dict()
            for z in range(1, 2*Lz, 2):
                operator[(x, 0, z)] = 'Z'
            logicals.append(operator)

        for y in range(2, 2*Ly, 2):
            operator = dict()
            for z in range(1, 2*Lz, 2):
                operator[(0, y, z)] = 'Z'
            logicals.append(operator)

        return logicals

    def stabilizer_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().stabilizer_representation(location, rotated_picture)

        if self.stabilizer_type(location) == 'face':
            axis, x, y, z = location
            representation['location'] = [x, y, z]

            if axis == 0:
                representation['params']['normal'] = [1, 0, 0]
            if axis == 1:
                representation['params']['normal'] = [0, 1, 0]

        return representation
