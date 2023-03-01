import itertools
from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class RhombicToricCode(StabilizerCode):
    """Toric code on periodic checkerboard lattice.

    Parameters
    ----------
    L_x : int
        Size of lattice along x direction. Must be even.
    L_y : Optional[int]
        Size of lattice along y direction. Must be even.
    L_z : Optional[int]
        Size of lattice along z direction. Must be even.

    Notes
    -----
    A checkerboard lattice is a cubic lattice for which there are two types of
    cells: colored and uncolored.
    For each colored cell, there is a 12-body stabilizer generator with support
    over the neighbour edges.
    For each uncolored cell, there is a stabilizer generator for each corner,
    each of which is a 3-body term with support over the 3 edges adjacent to
    both that corner vertex and the cell.
    """
    dimension = 3
    deformation_names = ['Checkerboard XZZX']

    @property
    def label(self) -> str:
        return 'Rhombic Toric {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: List[Tuple] = []
        Lx, Ly, Lz = self.size

        # Cubes
        ranges = [range(1, 2*Lx, 2), range(1, 2*Ly, 2), range(1, 2*Lz, 2)]
        for x, y, z in itertools.product(*ranges):
            if (x + y + z) % 4 == 1:
                coordinates.append((x, y, z))

        # Triangles
        ranges = [
            range(4), range(0, 2*Lx, 2), range(0, 2*Ly, 2), range(0, 2*Lz, 2)
        ]
        for axis, x, y, z in itertools.product(*ranges):
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
                np.add([x, y, z], d) % (2*np.array(self.size))
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

        # Sheet of X operators normal to the x direction
        operator: Operator = dict()
        for y in range(2*Ly):
            for z in range(2*Lz):
                if (y + z) % 2 == 1:
                    operator[(0, y, z)] = 'X'
        logicals.append(operator)

        # Sheet of X operators normal to the y direction
        operator = dict()
        for x in range(2*Lx):
            for z in range(2*Lz):
                if (x + z) % 2 == 1:
                    operator[(x, 0, z)] = 'X'
        logicals.append(operator)

        # Sheet of X operators normal to the z direction
        operator = dict()
        for x in range(2*Lx):
            for y in range(2*Ly):
                if (x + y) % 2 == 1:
                    operator[(x, y, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """The 3 logical Z operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Line of parallel Z operators along the x direction
        operator: Operator = dict()
        for x in range(0, 2*Lx, 2):
            operator[(x, 1, 0)] = 'Z'
        logicals.append(operator)

        # Line of parallel Z operators along the y direction
        operator = dict()
        for y in range(0, 2*Ly, 2):
            operator[(1, y, 0)] = 'Z'
        logicals.append(operator)

        # Line of parallel Z operators along the z direction
        operator = dict()
        for z in range(0, 2*Lz, 2):
            operator[(0, 1, z)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location, rotated_picture=False, json_file=None
    ) -> Dict:
        representation = super().stabilizer_representation(
            location, rotated_picture, json_file
        )

        if self.stabilizer_type(location) == 'triangle':
            axis, x, y, z = location
            representation['location'] = [x, y, z]

            delta_1 = [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]
            delta_2 = [[1, 1, -1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1]]

            delta = delta_1 if ((x + y + z) % 4 == 0) else delta_2

            a = 1.
            if rotated_picture:
                a = 1.

            dx, dy, dz = tuple(a * np.array(delta[axis]))

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
