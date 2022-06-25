from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple[int, int, int], str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple[int, int, int]]  # List of locations


class Toric3DCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Toric {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates = []
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
        coordinates = []
        Lx, Ly, Lz = self.size

        # Vertices
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in xy plane
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in yz plane
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in xz plane
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int, int]) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y, z = location
        if x % 2 == 0 and y % 2 == 0:
            return 'vertex'
        else:
            return 'face'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if self.stabilizer_type(location) == 'vertex':
            pauli = 'Z'
        else:
            pauli = 'X'

        deformed_pauli = {'X': 'Z', 'Z': 'X'}[pauli]

        x, y, z = location

        if self.stabilizer_type(location) == 'vertex':
            delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        else:
            # Face in xy-plane.
            if z % 2 == 0:
                delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
            # Face in yz-plane.
            elif (x % 2 == 0):
                delta = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
            # Face in zx-plane.
            elif (y % 2 == 0):
                delta = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            Lx, Ly, Lz = self.size
            qubit_location = ((x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly), (z + d[2]) % (2*Lz))

            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

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
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def get_logicals_x(self) -> Operator:
        """The 3 logical X operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 0, 0)] = 'X'
        logicals.append(operator)

        # X operators along y edges in y direction.
        operator = dict()
        for y in range(1, 2*Ly, 2):
            operator[(0, y, 0)] = 'X'
        logicals.append(operator)

        # X operators along z edges in z direction
        operator = dict()
        for z in range(1, 2*Lz, 2):
            operator[(0, 0, z)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        """Get the 3 logical Z operators."""
        Lx, Ly, Lz = self.size
        logicals = []

        # Z operators on x edges forming surface normal to x (yz plane).
        operator = dict()
        for y in range(0, 2*Ly, 2):
            for z in range(0, 2*Lz, 2):
                operator[(1, y, z)] = 'Z'
        logicals.append(operator)

        # Z operators on y edges forming surface normal to y (zx plane).
        operator = dict()
        for z in range(0, 2*Lz, 2):
            for x in range(0, 2*Lx, 2):
                operator[(x, 1, z)] = 'Z'
        logicals.append(operator)

        # Z operators on z edges forming surface normal to z (xy plane).
        operator = dict()
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                operator[(x, y, 1)] = 'Z'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().stabilizer_representation(location, rotated_picture)

        x, y, z = location
        if not rotated_picture and self.stabilizer_type(location) == 'face':
            if z % 2 == 0:  # xy plane
                representation['params']['normal'] = [0, 0, 1]
            elif x % 2 == 0:  # yz plane
                representation['params']['normal'] = [1, 0, 0]
            else:  # xz plane
                representation['params']['normal'] = [0, 1, 0]

        if rotated_picture and self.stabilizer_type(location) == 'face':
            if z % 2 == 0:
                representation['params']['normal'] = [0, 0, 1]
            elif x % 2 == 0:
                representation['params']['normal'] = [1, 0, 0]
            else:
                representation['params']['normal'] = [0, 1, 0]

        return representation
