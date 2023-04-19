from typing import Tuple, Dict, List
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class Color666PlanarCode(StabilizerCode):
    """2D color code on 6,6,6 lattice with open boundaries.

    The overall shape of the lattice is roughly a triangle.

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

    @property
    def label(self) -> str:
        return 'Color 6.6.6 Planar {}x{}'.format(*self.size)

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

        for x in range(2, 12*Lx+4, 1):
            for y in range(0, min(2*x + 1, 12*Lx - 2*x + 3), 1):
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
            qubit_location = (location[0] + d[0], location[1] + d[1])
            x, y = qubit_location

            if x >= 0 and 0 <= y <= min(2*x + 1, 12*Lx - 2*x + 3):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location: Tuple) -> str:
        x, y = location

        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        # X operators in x direction.
        operator: Operator = dict()
        for x in range(0, 12*Lx+4, 2):
            if self.is_qubit((x, 0)):
                operator[(x, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        # Z operators x direction.
        operator: Operator = dict()
        for x in range(0, 12*Lx+4, 2):
            if self.is_qubit((x, 0)):
                operator[(x, 0)] = 'Z'
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

        if y == 2*x:
            rep["params"]["vertices"] = [[-1, -2], [1, -2], [2, 0], [1, 2]]

        if y == 12*Lx - 2*x:
            rep["params"]["vertices"] = [[-1, -2], [1, -2], [-1, 2], [-2, 0]]

        if y == 0:
            rep["params"]["vertices"] = [[2, 0], [1, 2], [-1, 2], [-2, 0]]

        return rep
