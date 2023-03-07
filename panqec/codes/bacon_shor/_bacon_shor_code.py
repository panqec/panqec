from typing import Tuple, Dict, List
from panqec.codes import SubsystemCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class BaconShorCode(SubsystemCode):
    dimension = 2
    deformation_names = []

    @property
    def label(self) -> str:
        return 'Bacon-Shor {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        for x in range(0, 2*Lx+1, 2):
            for y in range(0, 2*Ly+1, 2):
                coordinates.append((x, y))

        return coordinates

    def get_gauge_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # XX on horizontal edges
        for x in range(1, 2*Lx+3, 2):
            for y in range(0, 2*Ly+1, 2):
                coordinates.append((x, y))

        # ZZ on vertical edges
        for x in range(0, 2*Lx+1, 2):
            for y in range(1, 2*Ly+3, 2):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Vertical X rectangular patch
        for x in range(1, 2*Lx+3, 2):
            coordinates.append((x, Ly, 0))

        # Horizontal Z rectangular patch
        for y in range(1, 2*Ly+3, 2):
            coordinates.append((Lx, y, 1))

        return coordinates

    def stabilizer_type(self, coordinates: Tuple) -> str:
        x, y, stab_type = coordinates
        if stab_type == 0:
            return 'vertical'  # X vertical patch
        else:
            return 'horizontal'  # Z horizontal patch

    def gauge_type(self, coordinates: Tuple) -> str:
        x, y = coordinates
        if x % 2 == 0:
            return 'vertical'  # Z vertical edge
        else:
            return 'horizontal'  # X horizontal edge

    def get_stabilizer_gauge_operators(self, stabilizer_coord) -> List[Tuple]:
        if self.stabilizer_type(stabilizer_coord) == 'horizontal':
            pauli = 'Z'
        else:
            pauli = 'X'

        Lx, Ly = self.size
        x, y, _ = stabilizer_coord

        gauge_coordinates = []
        if pauli == 'X':
            for gauge_y in range(0, 2*Ly+1, 2):
                gauge_coordinates.append((x, gauge_y))
        if pauli == 'Z':
            for gauge_x in range(0, 2*Lx+1, 2):
                gauge_coordinates.append((gauge_x, y))

        return gauge_coordinates

    def get_gauge_operator(self, location) -> Operator:
        if not self.is_gauge(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a gauge operator")

        if self.gauge_type(location) == 'horizontal':
            pauli = 'X'
        else:
            pauli = 'Z'

        Lx, Ly = self.size
        x, y = location

        operator: Operator = dict()

        if pauli == 'X':
            operator[((x-1) % (2*Lx+2), y)] = pauli
            operator[((x+1) % (2*Lx+2), y)] = pauli
        else:
            operator[(x, (y-1) % (2*Ly+2))] = pauli
            operator[(x, (y+1) % (2*Ly+2))] = pauli

        return operator

    def qubit_axis(self, location: Tuple) -> str:
        return 'none'

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        # Vertical column of Xs
        operator: Operator = dict()
        for y in range(0, 2*Ly+1, 2):
            operator[(0, y)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals: List[Operator] = []

        # Horizontal row of Zs
        operator: Operator = dict()
        for x in range(0, 2*Lx+1, 2):
            operator[(x, 0)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(
            location, rotated_picture, json_file=json_file
        )

        Lx, Ly = self.size
        x, y, _ = location
        rep['location'] = [x, y]

        if self.stabilizer_type(location) == 'horizontal':
            rep['params']['w'] = 2*Lx
        else:
            rep['params']['h'] = 2*Ly

        return rep
