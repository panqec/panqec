from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class Color488Code(StabilizerCode):
    """2D color code on periodic 4,8,8 lattice.

    Parameters
    ----------
    L_x : int
        Number of unit cells in x direction.
    L_y : Optional[int]
        Number of unit cells in y direction.

    Notes
    -----
    The 4,8,8 lattice is a tessallation of red squares (4),
    green octagons (8) and blue octagons (8).
    Qubits live on the vertices.
    For each colored shape, there are two stabilizer generators,
    one of all X over the qubits on its corners,
    and one of all Z over the qubits on its corners.

    See `EC zoo article <https://errorcorrectionzoo.org/c/color>`_
    for further reading.
    """
    dimension = 2
    deformation_names = ['XXZZ']

    @property
    def label(self) -> str:
        return 'Color 4.8.8 {}x{}'.format(*self.size)

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

        for x in range(0, 8*Lx+4, 4):
            for y in range(0, 8*Ly+4, 4):
                coordinates.append((x, y, 0))
                coordinates.append((x, y, 1))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        x, y, pauli = location

        if (x + y) % 8 == 0:
            stab_type = 'square-red'
        elif x % 8 == 4:
            stab_type = 'octahedron-blue'
        else:
            stab_type = 'octahedron-green'

        stab_type += '-' + {0: 'x', 1: 'z'}[pauli]

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

        stab_type = self.stabilizer_type(location)

        if 'square' in stab_type:
            delta = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
        else:
            delta = [(1, -3), (3, -1), (3, 1), (1, 3),
                     (-1, 3), (-3, 1), (-3, -1), (-1, -3)]

        operator: Operator = dict()
        for d in delta:
            qubit_location = ((location[0] + d[0]) % (8*Lx),
                              (location[1] + d[1]) % (8*Ly))
            x, y = qubit_location

            operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location: Tuple) -> str:
        x, y = location

        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        operator: Operator = dict()
        for y in range(3, 8*Ly+1, 2):
            if self.is_qubit((3, y)):
                operator[(3, y)] = 'X'
        logicals.append(operator)

        operator = dict()
        for y in range(1, 8*Lx+4, 2):
            if self.is_qubit((7, y)):
                operator[(7, y)] = 'X'
        logicals.append(operator)

        operator = dict()
        for x in range(3, 8*Lx+1, 2):
            if self.is_qubit((x, 5)):
                operator[(x, 5)] = 'X'
        logicals.append(operator)

        operator = dict()
        for x in range(1, 8*Ly+4, 2):
            if self.is_qubit((x, 1)):
                operator[(x, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        operator: Operator = dict()
        for x in range(3, 8*Lx+1, 2):
            if self.is_qubit((x, 5)):
                operator[(x, 5)] = 'Z'
        logicals.append(operator)

        operator = dict()
        for x in range(1, 8*Ly+4, 2):
            if self.is_qubit((x, 1)):
                operator[(x, 1)] = 'Z'
        logicals.append(operator)

        operator = dict()
        for y in range(3, 8*Ly+1, 2):
            if self.is_qubit((3, y)):
                operator[(3, y)] = 'Z'
        logicals.append(operator)

        operator = dict()
        for y in range(1, 8*Lx+4, 2):
            if self.is_qubit((7, y)):
                operator[(7, y)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(location, rotated_picture)

        Lx, Ly = self.size
        x, y, _ = location

        # We remove the last part of the location (indexing X or Z)
        rep['location'] = (x, y)

        if '-x' in self.stabilizer_type(location):
            a = 0.5
        else:
            a = 1

        if 'octahedron' in self.stabilizer_type(location):
            if x == 0:
                rep['params']['vertices'] = [[0, -3*a], [a, -3*a], [3*a, -a],
                                             [3*a, a], [a, 3*a], [0, 3*a]]
            elif x == 8*Lx:
                rep['params']['vertices'] = [[0, -3*a], [0, 3*a], [-a, 3*a],
                                             [-3*a, a], [-3*a, -a], [-a, -3*a]]
            elif y == 0:
                rep['params']['vertices'] = [[3*a, 0], [3*a, a], [a, 3*a],
                                             [-a, 3*a], [-3*a, a], [-3*a, 0]]
            elif y == 8*Ly:
                rep['params']['vertices'] = [[a, -3*a], [3*a, -a], [3*a, 0],
                                             [-3*a, 0], [-3*a, -a], [-a, -3*a]]
        else:
            if x == 0:
                if y == 0:
                    rep['params']['vertices'] = [[0, 0], [a, 0],
                                                 [a, a], [0, a]]
                elif y == 8*Ly:
                    rep['params']['vertices'] = [[0, 0], [a, 0],
                                                 [a, -a], [0, -a]]
                else:
                    rep['params']['vertices'] = [[0, a], [a, a],
                                                 [a, -a], [0, -a]]
            elif x == 8*Lx:
                if y == 0:
                    rep['params']['vertices'] = [[0, 0], [0, a],
                                                 [-a, a], [-a, 0]]
                elif y == 8*Ly:
                    rep['params']['vertices'] = [[0, 0], [0, -a],
                                                 [-a, -a], [-a, 0]]
                else:
                    rep['params']['vertices'] = [[-a, a], [0, a],
                                                 [0, -a], [-a, -a]]
            elif y == 0:
                rep['params']['vertices'] = [[-a, 0], [a, 0],
                                             [a, a], [-a, a]]

            elif y == 8*Ly:
                rep['params']['vertices'] = [[-a, 0], [a, 0],
                                             [a, -a], [-a, -a]]

        return rep

    def get_deformation(
        self, location: Tuple,
        deformation_name: str,
        **kwargs
    ) -> Dict:

        x, y = location
        if deformation_name == 'XXZZ':
            undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

            if (x + y - 4) % 4 == 0:
                deformation = deformed_dict
            else:
                deformation = undeformed_dict

        else:
            raise ValueError(f"The deformation {deformation_name}"
                             "does not exist")

        return deformation
