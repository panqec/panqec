from typing import List, Tuple, Dict
from panqec.codes import SubsystemCode

Operator = Dict[Tuple, str]  # Coordinate to pauli ('X', 'Y' or 'Z')


class BaconShorCode(SubsystemCode):

    dimension = 2

    def __init__(self, L_x: int, L_y: int):
        super().__init__(L_x, L_y)

    @property
    def label(self) -> str:
        return 'Bacon-Shor {}x{}'.format(*self.size)

    def get_gauge_coordinates(self) -> List[Tuple]:
        coordinates: List[Tuple] = []
        L_x, L_y = self.size
        for i_x in range(L_x - 1):
            for i_y in range(L_y):
                coordinates.append((2*i_x + 1, 2*i_y, 0))
        for i_x in range(L_x):
            for i_y in range(L_y - 1):
                coordinates.append((2*i_x, 2*i_y + 1, 0))
        return coordinates

    def get_gauge(self, location: Tuple) -> Operator:
        operator: Operator = dict()
        x, y, _ = location
        if x % 2 == 1:
            operator[(x - 1, y, 0)] = 'X'
            operator[(x + 1, y, 0)] = 'X'
        else:
            operator[(x, y - 1, 0)] = 'Z'
            operator[(x, y + 1, 0)] = 'Z'
        return operator

    def get_stabilizers_from_gauges(self) -> Dict[Tuple, List[Tuple]]:
        mapping: Dict[Tuple, List[Tuple]] = dict()
        return mapping

    def get_stabilizer_coordinates(self) -> List[Tuple]:
        coordinates: List[Tuple] = []
        L_x, L_y = self.size

        # Coordinate of each stabilizer is at the centroid of the strip.
        # X stabilizers.
        for i_x in range(L_x - 1):
            coordinates.append((i_x + 1, L_y, 0))

        # Z stabilizers.
        for i_y in range(L_y - 1):
            coordinates.append((L_x, i_y + 1, 1))
        return coordinates

    def get_qubit_coordinates(self) -> List[Tuple]:
        coordinates: List[Tuple] = []
        L_x, L_y = self.size
        for i_x in range(L_x):
            for i_y in range(L_y):
                coordinates.append((2*i_x, 2*i_y, 0))
        return coordinates

    def get_stabilizer(
        self, location: Tuple, deformed_axis: str = None
    ) -> Operator:
        operator: Operator = dict()
        x, y, layer = location
        if layer == 0:
            pass
        else:
            pass
        return operator

    def stabilizer_type(self, location: Tuple) -> str:
        x, y, layer = location
        if layer == 0:
            return 'x'
        else:
            return 'z'

    def get_logicals_x(self) -> List[Operator]:
        logicals: List[Operator] = []
        return logicals

    def get_logicals_z(self) -> List[Operator]:
        logicals: List[Operator] = []
        return logicals
