from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple[int, int, int], str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple[int, int, int]]  # List of locations


class Planar3DCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Planar {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates = []
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
        coordinates = []
        Lx, Ly, Lz = self.size

        # Vertices
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Faces in xy plane
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Faces in yz plane
        for x in range(2, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Faces in xz plane
        for x in range(1, 2*Lx+1, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz-1, 2):
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
            delta = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
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
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location) -> int:
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
        """The unique logical X operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, 2*Lx+1, 2):
            operator[(x, 0, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        """The unique logical Z operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for y in range(0, 2*Ly, 2):
            for z in range(0, 2*Lz, 2):
                operator[(1, y, z)] = 'Z'
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