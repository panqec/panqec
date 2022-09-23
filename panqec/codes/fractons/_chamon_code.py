from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations


class ChamonCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Chamon {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        print(self.size)
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(0, 2*Lx, 1):
            for y in range(0, 2*Ly, 1):
                for z in range(0, 2*Lz, 1):
                    if (x + y + z) % 2 == 0:
                        coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(0, 2*Lx, 1):
            for y in range(0, 2*Ly, 1):
                for z in range(0, 2*Lz, 1):
                    if (x + y + z) % 2 == 1:
                        coordinates.append((x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        return 'vertex'

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        Lx, Ly, Lz = self.size
        x, y, z = location

        delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1),
                 (0, 0, 1)]
        paulis = ['Y', 'Y', 'Z', 'Z', 'X', 'X']

        operator: Operator = dict()
        for i, d in enumerate(delta):
            qubit_location = ((x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly),
                              (z + d[2]) % (2*Lz))

            operator[qubit_location] = paulis[i]

        return operator

    def qubit_axis(self, location):
        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly, Lz = self.size
        logicals = []

        operator: Operator = dict()
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                operator[(x, y, 0)] = 'X'
        logicals.append(operator)

        operator: Operator = dict()
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                operator[(x, y, 0)] = 'X'
        logicals.append(operator)

        operator: Operator = dict()
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                operator[(x, y, 1)] = 'X'
        logicals.append(operator)

        operator: Operator = dict()
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                operator[(x, y, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly, Lz = self.size
        logicals = []

        operator: Operator = dict()
        for x in range(0, 2*Lx, 2):
            for z in range(0, 2*Lz, 2):
                operator[(x, 0, z)] = 'Z'
        logicals.append(operator)

        operator: Operator = dict()
        for x in range(1, 2*Lx, 2):
            for z in range(0, 2*Lz, 2):
                operator[(x, 1, z)] = 'Z'
        logicals.append(operator)

        operator: Operator = dict()
        for x in range(0, 2*Lx, 2):
            for z in range(1, 2*Lz, 2):
                operator[(x, 1, z)] = 'Z'
        logicals.append(operator)

        operator: Operator = dict()
        for x in range(1, 2*Lx, 2):
            for z in range(1, 2*Lz, 2):
                operator[(x, 0, z)] = 'Z'
        logicals.append(operator)

        return logicals
