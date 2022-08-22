from typing import Tuple, Dict, List
import itertools
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations


class Color3DCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Color {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(0, 6*Lx+5, 2):
            for y in range(0, 2*Ly+1, 2):
                for z in range(0, 4*Lz+3, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        # Cells
        for y in range(1, 2*Ly, 2):
            z0 = 2 if (y % 4 == 1) else 4
            for z in range(z0, 4*Lz+2, 4):
                x0 = 3 + z % 6
                for x in range(x0, 6*Lx+3, 6):
                    coordinates.append((x, y, z))

        # # Faces
        # for y in range(1, 2*Ly, 2):
        #     z0 = 1 if (y % 4 == 1) else 3
        #     for z in range(z0, 4*Lz+2, 4):
        #         x0 = (4 - z // 2 - 3*(y % 4 != 1)) % 6
        #         x0 = x0 if x0 else 6
        #         for x in range(x0, 6*Lx+3, 6):
        #             coordinates.append((x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y, z = location

        color = {3: 'cell-red',
                 5: 'cell-yellow',
                 7: 'cell-blue',
                 1: 'cell-green'}

        return color[x % 8]

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if 'cell' in self.stabilizer_type(location):
            pauli = 'X'
        else:
            pauli = 'Z'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        x, y, z = location

        if 'cell' in self.stabilizer_type(location):
            delta = []

            coord = itertools.product([-3, -1, 1, 3], [-1, 1], [-2, 0, 2])
            for dx, dy, dz in coord:
                delta.append((dx, dy, dz))

        operator: Operator = dict()
        for d in delta:
            qubit_location = ((x + d[0]), (y + d[1]), (z + d[2]))

            if self.is_qubit(qubit_location):
                is_deformed = (
                    self.qubit_axis(qubit_location) == deformed_axis
                )
                operator[qubit_location] = (deformed_pauli if is_deformed
                                            else pauli)

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        """The 3 logical X operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """Get the 3 logical Z operators."""
        Lx, Ly, Lz = self.size
        logicals = []

        return logicals
