from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode
import numpy as np

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class Color666ToricCode(StabilizerCode):
    """2D color code on periodic 6,6,6 lattice.

    The overall shape of the lattice is roughly a rhombus.

    Parameters
    ----------
    L_x : int
        Number of unit cells in x direction.
    L_y : Optional[int]
        Number of unit cells in y direction.

    Notes
    -----
    The 6,6,6 lattice is a tessallation of red hexagons (6),
    green hexagons (6) and blue hexagons (6).
    Qubits live on the vertices.
    For each colored shape, there are two stabilizer generators,
    one of all X over the qubits on its corners,
    and one of all Z over the qubits on its corners.

    See `EC zoo article <https://errorcorrectionzoo.org/c/color>`_
    for further reading.
    """
    dimension = 2
    deformation_names = ['X3Z3']

    @property
    def label(self) -> str:
        return 'Color 6.6.6 Toric {}x{}'.format(*self.size)

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

        for x in range(2, 9*Lx, 3):
            for y in range(2+2*(x-2)//3, 12*Ly+2*(x-2)//3+1, 4):
                coordinates.append((x, y, 0))
                coordinates.append((x, y, 1))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        x, y, pauli = location
        stab_type = 'face'

        if (x % 6 == 2 and y % 12 == 2) or (x % 6 == 5 and y % 12 == 8):
            stab_type += '-green'
        elif (x % 6 == 5 and y % 12 == 0) or (x % 6 == 2 and y % 12 == 6):
            stab_type += '-blue'
        else:
            stab_type += '-red'

        if pauli == 0:
            stab_type += '-x'
        else:
            stab_type += '-z'

        return stab_type

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        Lx, Ly = self.size

        if location[-1] == 0:
            pauli = 'X'
        else:
            pauli = 'Z'

        delta: List[Tuple] = [(-1, -2), (1, -2), (2, 0),
                              (1, 2), (-1, 2), (-2, 0)]

        operator: Operator = dict()
        for d in delta:
            x = (location[0] + d[0]) % (9*Lx)
            y = (location[1] + d[1])

            if (location[0] + d[0] >= 9*Lx):
                y -= 6*Lx

            if (y >= 12*Ly+2*(x-2)//3):
                y -= 12*Ly

            if (x == 0 and y == -2):  # Qubit which was on a corner
                x = 0
                y = 12*Ly - 2

            operator[(x, y)] = pauli

        return operator

    def qubit_axis(self, location: Tuple) -> str:
        x, y = location

        return 'x'

    def get_deformation(
        self, location: Tuple,
        deformation_name: str,
        **kwargs
    ) -> Dict:

        x, y = location

        if deformation_name == 'X3Z3':
            undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

            x_to_y = {0: 2, 10: 2, 1: 0, 3: 0, 4: 6, 6: 6, 7: 4, 9: 4}

            if x_to_y[x % 12] == y % 8:
                deformation = deformed_dict
            else:
                deformation = undeformed_dict
        else:
            raise ValueError(f"The deformation {deformation_name}"
                             "does not exist")

        return deformation

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        operator: Operator = dict()
        for x in range(8, 9*Lx, 9):
            y = 12*Ly - 6 - 6 * (x-8) // 9
            operator[(x-4, y+4)] = 'X'
            operator[(x-2, y+4)] = 'X'

            if self.is_qubit((x+1, y+2)):
                operator[(x+1, y+2)] = 'X'
                operator[(x+2, y)] = 'X'
            else:
                operator[(0, 2)] = 'X'
                operator[(1, 0)] = 'X'
        logicals.append(operator)

        operator = dict()
        for x in range(5, 9*Lx, 9):
            y = 12*Ly - 4 - 6 * (x-5) // 9
            operator[(x-5, y+2)] = 'X'
            operator[(x-4, y)] = 'X'

            operator[(x-1, y-2)] = 'X'
            operator[(x+1, y-2)] = 'X'
        logicals.append(operator)

        operator = dict()
        for y in range(8, 12*Ly, 12):
            x = 5
            operator[(x-1, y+2)] = 'X'
            operator[(x-2, y)] = 'X'
            operator[(x-2, y-4)] = 'X'
            operator[(x-1, y-6)] = 'X'
        logicals.append(operator)

        operator = dict()
        for y in range(8, 12*Ly, 12):
            x = 5
            operator[(x-2, y)] = 'X'
            operator[(x-1, y-2)] = 'X'
            operator[(x-1, y-6)] = 'X'
            operator[(x-2, y-8)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        operator: Operator = dict()
        for y in range(8, 12*Ly, 12):
            x = 5
            operator[(x-1, y+2)] = 'Z'
            operator[(x-2, y)] = 'Z'
            operator[(x-2, y-4)] = 'Z'
            operator[(x-1, y-6)] = 'Z'
        logicals.append(operator)

        operator = dict()
        for y in range(8, 12*Ly, 12):
            x = 5
            operator[(x-2, y)] = 'Z'
            operator[(x-1, y-2)] = 'Z'
            operator[(x-1, y-6)] = 'Z'
            operator[(x-2, y-8)] = 'Z'
        logicals.append(operator)

        operator = dict()
        for x in range(8, 9*Lx, 9):
            y = 12*Ly - 6 - 6 * (x-8) // 9
            operator[(x-4, y+4)] = 'Z'
            operator[(x-2, y+4)] = 'Z'

            if self.is_qubit((x+1, y+2)):
                operator[(x+1, y+2)] = 'Z'
                operator[(x+2, y)] = 'Z'
            else:
                operator[(0, 2)] = 'Z'
                operator[(1, 0)] = 'Z'
        logicals.append(operator)

        operator = dict()
        for x in range(5, 9*Lx, 9):
            y = 12*Ly - 4 - 6 * (x-5) // 9
            operator[(x-5, y+2)] = 'Z'
            operator[(x-4, y)] = 'Z'

            operator[(x-1, y-2)] = 'Z'
            operator[(x+1, y-2)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(location, rotated_picture)

        x, y, _ = location

        # We remove the last part of the location (indexing X or Z)
        rep['location'] = (x, y)

        # Scaling factor for the X hexagons
        if '-x' in self.stabilizer_type(location):
            a = 0.5
        else:
            a = 1

        vertices = np.array(rep['params']['vertices']) * a
        rep['params']['vertices'] = vertices.tolist()

        return rep
