import itertools
from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class CheckerboardCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Checkerboard {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Cubes
        ranges = [range(2), range(1, 2*Lx, 2), range(1, 2*Ly, 2),
                  range(1, 2*Lz, 2)]
        for stab_type, x, y, z in itertools.product(*ranges):
            if (x + y + z) % 4 == 3:
                coordinates.append((stab_type, x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int, int]) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if location[0] == 0:
            return 'cube-x'
        else:
            return 'cube-z'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if self.stabilizer_type(location) == 'cube-x':
            pauli = 'X'
        else:
            pauli = 'Z'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        _, x, y, z = location
        delta = itertools.product([-1, 1], repeat=3)

        Lx, Ly, Lz = self.size
        operator = dict()
        for d in delta:
            qubit_location = ((x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly), (z + d[2]) % (2*Lz))
            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location):
        return 0

    def get_logicals_x(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        operator = dict()
        operator[(0, 0, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        operator = dict()
        operator[(0, 0, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location, rotated_picture=False
    ) -> Dict:
        representation = super().stabilizer_representation(location,
                                                           rotated_picture)

        stab_type, x, y, z = location
        representation['location'] = [x, y, z]

        return representation
