import itertools
from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class HaahCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Haah Code {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Haah's code has two qubits per site, i=0 and i=1
        for i in range(2):
            for x in range(0, 2*Lx, 2):
                for y in range(0, 2*Ly, 2):
                    for z in range(0, 2*Lz, 2):
                        coordinates.append((i, x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        # i=0: X stabilizer ; i=1: Z stabilizer
        for i in range(2):
            for x in range(1, 2*Lx, 2):
                for y in range(1, 2*Ly, 2):
                    for z in range(1, 2*Lz, 2):
                        coordinates.append((i, x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int, int, int]) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if location[0] == 0:
            return 'x-cube'
        else:
            return 'z-cube'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        stab_type = self.stabilizer_type(location)

        if stab_type == 'x-cube':
            pauli = 'X'
        else:
            pauli = 'Z'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        delta = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]

        if stab_type == 'z-cube':
            double_op = ['IZ', 'ZI', 'IZ', 'II',
                         'ZI', 'ZZ', 'ZI', 'IZ']
        else:
            double_op = ['IX', 'XI', 'IX', 'XX',
                         'XI', 'II', 'XI', 'IX']

        _, x, y, z = location
        Lx, Ly, Lz = self.size
        operator = dict()

        for d, op in zip(delta, double_op):
            qubit_location = [(0, (x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly), (z + d[2]) % (2*Lz)),
                              (1, (x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly), (z + d[2]) % (2*Lz))]

            # TODO: change that
            is_deformed = False

            for i in range(2):
                if op[i] != 'I':
                    operator[qubit_location[i]] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location):
        return 'x'

    def get_logicals_x(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        return logicals

    def get_logicals_z(self) -> Operator:
        Lx, Ly, Lz = self.size
        logicals = []

        return logicals

    def stabilizer_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().stabilizer_representation(location, rotated_picture)

        representation['location'] = location[1:]

        return representation

    def qubit_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().qubit_representation(location, rotated_picture)

        i, x, y, z = location

        d = representation['params']['radius']
        if i == 0:
            representation['location'] = [x-d, y, z]
        else:
            representation['location'] = [x+d, y, z]

        return representation
