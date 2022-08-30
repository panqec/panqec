from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode
import numpy as np

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class Color2DCode(StabilizerCode):
    dimension = 2

    @property
    def label(self) -> str:
        return 'Planar {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        coordinates = []

        stab_coordinates = self.get_stabilizer_coordinates()

        for location in stab_coordinates:
            qubit_coords = list(self.get_stabilizer(location).keys())
            for coord in qubit_coords:
                if coord not in coordinates:
                    coordinates.append(coord)

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        for x in range(2, 4*Lx, 1):
            for y in range(0, min(2*x + 1, 4*Lx - 2*x - 1), 1):
                if (x % 6 == 2 and y % 4 == 0) or (x % 6 == 5 and y % 4 == 2):
                    coordinates.append((x, y, 0))
                    coordinates.append((x, y, 1))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        x, y, pauli = location
        stab_type = 'face'

        if (x % 6 == 2 and y % 12 == 0) or (x % 6 == 5 and y % 12 == 6):
            stab_type += '-green'
        elif (x % 6 == 5 and y % 12 == 2) or (x % 6 == 2 and y % 12 == 8):
            stab_type += '-blue'
        else:
            stab_type += '-red'

        if pauli == 0:
            stab_type += '-x'
        else:
            stab_type += '-z'

        return stab_type

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        Lx, Ly = self.size

        if location[-1] == 0:
            pauli = 'X'
        else:
            pauli = 'Z'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        delta: List[Tuple] = [(-1, -2), (1, -2), (2, 0),
                              (1, 2), (-1, 2), (-2, 0)]

        operator: Operator = dict()
        for d in delta:
            qubit_location = (location[0] + d[0], location[1] + d[1])
            x, y = qubit_location

            if x >= 0 and 0 <= y <= min(2*x + 1, 4*Lx - 2*x - 1):
                is_deformed = (
                    self.qubit_axis(qubit_location) == deformed_axis
                )
                operator[qubit_location] = (deformed_pauli if is_deformed
                                            else pauli)

        return operator

    def qubit_axis(self, location: Tuple) -> str:
        x, y = location

        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        # X operators in x direction.
        operator: Operator = dict()
        for x in range(0, 4*Lx, 2):
            if self.is_qubit((x, 0)):
                operator[(x, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        # Z operators x direction.
        operator: Operator = dict()
        for x in range(0, 4*Lx, 2):
            if self.is_qubit((x, 0)):
                operator[(x, 0)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(location, rotated_picture)

        # We remove the last part of the location (indexing X or Z)
        rep['location'] = (location[0], location[1])

        return rep