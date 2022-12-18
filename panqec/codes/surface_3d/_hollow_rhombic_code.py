import itertools
from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class HollowRhombicCode(StabilizerCode):
    dimension = 3
    deformation_names = ['Checkerboard XZZX']

    @property
    def label(self) -> str:
        return 'Hollow Rhombic {}x{}x{}'.format(*self.size)

    def _is_in_hole(self, x, y, z):
        Lx, Ly, Lz = self.size

        return ((x > 2 and x < 2*Lx-2)
                and (y >= 3 and y < 2*Ly-4)
                and (z >= 3 and z < 2*Lz-4))

    def _is_m_boundary(self, x, y, z):
        Lx, Ly, Lz = self.size

        return (y >= 2*Ly-2 or y <= 0 or
                2*Ly-5 <= y <= 2*Ly-4 or 2 <= y <= 3 or
                2*Lz-5 <= z <= 2*Lz-4 or 2 <= z <= 3 or
                2 <= x <= 3 or 2*Lx-3 <= x <= 2*Lx-2)

    def _is_e_boundary(self, x, y, z):
        Lx, Ly, Lz = self.size

        return (z >= 2*Lz-2 or z <= 0)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    if not self._is_in_hole(x, y, z):
                        coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(2, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(0, 2*Lz, 2):
                    if not self._is_in_hole(x, y, z):
                        coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz-1, 2):
                    if not self._is_in_hole(x, y, z):
                        coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: List[Tuple] = []
        Lx, Ly, Lz = self.size

        # Cubes
        ranges = [range(1, 2*Lx, 2), range(-1, 2*Ly, 2), range(1, 2*Lz-1, 2)]
        for x, y, z in itertools.product(*ranges):
            on_edge = (
                (y == -1 and z == -1) or
                (y == -1 and z == 2*Lz-1) or
                (y == 2*Ly-1 and z == -1) or
                (y == 2*Ly-1 and z == 2*Lz-1)
            )
            if (((x + y + z) % 4 == 1)
                    and not on_edge
                    and not np.all([
                        self._is_in_hole(x+d[0], y+d[1], z+d[2])
                        for d in itertools.product([-1, 1], repeat=3)
                    ])):
                coordinates.append((x, y, z))

        # Triangles
        ranges = [
            range(4), range(2, 2*Lx, 2), range(0, 2*Ly, 2), range(0, 2*Lz, 2)
        ]
        for axis, x, y, z in itertools.product(*ranges):
            location = (axis, x, y, z)
            stab = list(self.get_stabilizer(location).keys())

            em_edge = (y, z) in itertools.product([0, 2*Ly-2], [0, 2*Lz-2])
            constant_z = np.all([loc[2] == stab[0][2] for loc in stab])

            excluded_triangle = (
                len(stab) <= 1 or (
                    self._is_m_boundary(x, y, z) and
                    len(stab) <= 2 and
                    (not self._is_e_boundary(x, y, z) or (
                        em_edge and not constant_z
                    ))
                )
            )

            if (not excluded_triangle
                    and not self._is_in_hole(x, y, z)):
                coordinates.append((axis, x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if len(location) == 4:
            return 'triangle'
        else:
            return 'cube'

    def get_stabilizer(self, location) -> Operator:
        if self.stabilizer_type(location) == 'cube':
            pauli = 'X'
        else:
            pauli = 'Z'

        if self.stabilizer_type(location) == 'cube':
            x, y, z = location
            delta = [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                     (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
                     (0, 1, 1), (0, -1, -1), (0, -1, 1), (0, 1, -1)]
        else:
            axis, x, y, z = location
            if (x + y + z) % 4 == 0:
                delta_axis = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                              [(-1, 0, 0), (0, -1, 0), (0, 0, 1)],
                              [(1, 0, 0), (0, -1, 0), (0, 0, -1)],
                              [(-1, 0, 0), (0, 1, 0), (0, 0, -1)]]
            else:
                delta_axis = [[(1, 0, 0), (0, 1, 0), (0, 0, -1)],
                              [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
                              [(1, 0, 0), (0, -1, 0), (0, 0, 1)],
                              [(-1, 0, 0), (0, 1, 0), (0, 0, 1)]]
            delta = delta_axis[axis]

        operator = dict()
        for d in delta:
            qubit_location = tuple(
                np.add([x, y, z], d)
            )

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        if (z % 2 == 0) and (x % 2 == 1) and (y % 2 == 0):
            axis = 'x'
        elif (z % 2 == 0) and (x % 2 == 0) and (y % 2 == 1):
            axis = 'y'
        elif (z % 2 == 1) and (x % 2 == 0) and (y % 2 == 0):
            axis = 'z'
        else:
            raise ValueError(
                f'Location {location} does not correspond to a qubit'
            )

        return axis

    def get_logicals_x(self) -> List[Operator]:
        """The 3 logical X operators."""

        Lx, Ly, Lz = self.size
        logicals: List[Operator] = []

        # Sheet of X operators normal to the z direction
        operator: Operator = dict()
        for x in range(2*Lx):
            for y in range(2*Ly):
                if self.is_qubit((x, y, 4)):
                    operator[(x, y, 4)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """The 3 logical Z operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Line of parallel Z operators along the z direction
        operator: Operator = dict()
        for z in range(0, 2*Lz, 2):
            operator[(2*Lx-1, 2*Ly-2, z)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location, rotated_picture=False, json_file=None
    ) -> Dict:
        representation = super().stabilizer_representation(
            location, rotated_picture, json_file
        )

        Lx, Ly, Lz = self.size

        if self.stabilizer_type(location) == 'cube':
            x, y, z = location
            if len(self.get_stabilizer(location)) <= 4:
                angle = np.pi/4 if rotated_picture else 0
                representation['object'] = 'rectangle'
                representation['params'] = {
                    "w": 1.5,
                    "h": 1.5,
                    "normal": [0, 1, 0],
                    "angle": angle
                }
                if y == -1 or y == 2*Ly-5:
                    representation['location'] = [x, y+0.9, z]
                elif y == 3 or y == 2*Ly-1:
                    representation['location'] = [x, y-0.9, z]
                elif z == -1 or z == 2*Lz-5:
                    representation['params']['normal'] = [0, 0, 1]
                    representation['location'] = [x, y, z+0.9]
                elif z == 2*Lz-1 or z == 3:
                    representation['params']['normal'] = [0, 0, 1]
                    representation['location'] = [x, y, z-0.9]
                elif x == 3:
                    representation['params']['normal'] = [1, 0, 0]
                    representation['location'] = [x-0.9, y, z]
                elif x == 2*Lx-3:
                    representation['params']['normal'] = [1, 0, 0]
                    representation['location'] = [x+0.9, y, z]

        elif self.stabilizer_type(location) == 'triangle':
            axis, x, y, z = location
            representation['location'] = [x, y, z]

            delta_1 = [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]
            delta_2 = [[1, 1, -1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1]]

            delta = delta_1 if ((x + y + z) % 4 == 0) else delta_2

            a = 1.
            dx, dy, dz = tuple(a * np.array(delta[axis]))

            if len(self.get_stabilizer(location)) <= 2:
                if (2 < x < 2*Lx-2 and 0 < z < 2*Lz-2 and
                        (y == 2 or y == 2*Ly-4)):
                    dy = 0
                if ((z == 0 and
                        (dz == -1 or (2 < x < 2*Lx-2 and 2 < y < 2*Lz-2))) or
                    (z == 2*Lz-2 and
                        (dz == 1 or (2 < x < 2*Lx-2 and 2 < y < 2*Ly-4)))):
                    dz = 0
                if ((2 < y < 2*Ly-4 and 0 < z < 2*Lz-2) and
                        ((x == 2 and dx == 1) or (x == 2*Lx-2 and dx == -1))):
                    dx = 0

            representation['params']['vertices'] = [[dx, 0, 0],
                                                    [0, dy, 0],
                                                    [0, 0, dz]]

        return representation

    def get_deformation(
        self, location: Tuple,
        deformation_name: str,
        **kwargs
    ) -> Dict:
        undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}

        if deformation_name == 'Checkerboard XZZX':
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}
        else:
            raise ValueError(f"The deformation {deformation_name}"
                             "does not exist")

        x, y, z = location

        if (self.qubit_axis(location) == 'z'
                and ((z % 4 == 3 and (x + y) % 4 == 2)
                     or (z % 4 == 1 and (x + y) % 4 == 0))):
            return deformed_dict
        else:
            return undeformed_dict
