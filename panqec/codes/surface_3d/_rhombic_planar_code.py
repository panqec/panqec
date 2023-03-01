import itertools
from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class RhombicPlanarCode(StabilizerCode):
    """Toric code on checkerboard lattice with open boundaries.

    Similar to :class:`panqec.codes.surface_3d.RhombicToricCode`
    but with smooth boundaries on planes orthogonal to the x and z directions
    and rough boundaries on planes orthogonal to the y direction.

    Parameters
    ----------
    L_x : int
        Size of lattice along x direction,
        which is actually the number of x-edges in the x direction.
    L_y : Optional[int]
        Size of lattice along y direction,
        which is actually the number of x-edges in the y direction.
    L_z : Optional[int]
        Size of lattice along z direction,
        which is actually the number of x-edges in the z direction.

    Notes
    -----
    Note that it is the same qubit lattice as that of
    :class:`panqec.codes.surface_3d.Planar3DCode`
    but with very different stabilizer generators.
    """
    dimension = 3
    deformation_names = ['Checkerboard XZZX']

    @property
    def label(self) -> str:
        return 'Rhombic Planar {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(2, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz-1, 2):
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
            on_rough_boundary = False
            if (((x + y + z) % 4 == 1 or on_rough_boundary) and not on_edge):
                coordinates.append((x, y, z))

        # Triangles
        ranges = [
            range(4), range(2, 2*Lx, 2), range(0, 2*Ly, 2), range(0, 2*Lz, 2)
        ]
        for axis, x, y, z in itertools.product(*ranges):
            edge_triangle = (
                (z == 2*Lz-2 and y == 2*Ly-2 and (
                    ((x + y + z) % 4 == 0 and axis == 0)
                    or ((x + y + z) % 4 != 0 and axis == 3)))
                or (z == 0 and y == 0 and (
                    ((x + y + z) % 4 == 0 and axis == 2)
                    or ((x + y + z) % 4 != 0 and axis == 1)))
                or (z == 2*Lz-2 and y == 0 and (
                    ((x + y + z) % 4 == 0 and axis == 1)
                    or ((x + y + z) % 4 != 0 and axis == 2)))
                or (z == 0 and y == 2*Ly-2 and (
                    ((x + y + z) % 4 == 0 and axis == 3)
                    or ((x + y + z) % 4 != 0 and axis == 0)))
            )
            rough_triangle = (
                (y == 0 and axis in [1, 2]) or
                (y == 2*Ly-2 and axis in [0, 3])
            )

            if not edge_triangle and not rough_triangle:
                coordinates.append((axis, x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if len(location) == 4:
            return 'triangle'
        else:
            return 'cube'

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

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
                if self.is_qubit((x, y, 0)):
                    operator[(x, y, 0)] = 'X'
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
            if y == -1 or y == 2*Ly-1 or z == -1 or z == 2*Lz-1:
                angle = np.pi/4 if rotated_picture else 0
                representation['object'] = 'rectangle'
                representation['params'] = {
                    "w": 1.5,
                    "h": 1.5,
                    "normal": [0, 1, 0],
                    "angle": angle
                }
                if y == -1:
                    representation['location'] = [x, y+0.9, z]
                elif y == 2*Ly-1:
                    representation['location'] = [x, y-0.9, z]
                elif z == -1:
                    representation['params']['normal'] = [0, 0, 1]
                    representation['location'] = [x, y, z+0.9]
                elif z == 2*Lz-1:
                    representation['params']['normal'] = [0, 0, 1]
                    representation['location'] = [x, y, z-0.9]

        elif self.stabilizer_type(location) == 'triangle':
            axis, x, y, z = location
            representation['location'] = [x, y, z]

            delta_1 = [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]
            delta_2 = [[1, 1, -1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1]]

            delta = delta_1 if ((x + y + z) % 4 == 0) else delta_2

            a = 1.
            dx, dy, dz = tuple(a * np.array(delta[axis]))

            if (y == 0 and dy == -1) or (y == 2*Ly-2 and dy == 1):
                dy = 0
            if (z == 0 and dz == -1) or (z == 2*Lz-2 and dz == 1):
                dz = 0

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
