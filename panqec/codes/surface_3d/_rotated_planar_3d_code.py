from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple[int, int, int], str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple[int, int, int]]  # List of locations


class RotatedPlanar3DCode(StabilizerCode):
    dimension = 3

    @property
    def label(self) -> str:
        return 'Rotated Planar {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Operator:
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(2, 2*Lz, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Vertices
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly+1, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        # Horizontal faces
        for x in range(0, 2*Lx+1, 2):
            for y in range(2, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))

        # Vertical faces
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(2, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple[int, int, int]) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y, z = location
        if (x + y) % 4 == 2 and z % 2 == 1:
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
            delta = [(-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
        else:
            # z-normal so face is xy-plane.
            if z % 2 == 1:
                delta = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]
            # x-normal so face is in yz-plane.
            elif (x + y) % 4 == 0:
                delta = [(-1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
            # y-normal so face is in zx-plane.
            elif (x + y) % 4 == 2:
                delta = [(-1, 1, 0), (1, -1, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                is_deformed = (self.qubit_axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        if location not in self.qubit_coordinates:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        if (z % 2 == 0):
            axis = 'z'
        elif (x + y) % 4 == 2:
            axis = 'x'
        elif (x + y) % 4 == 0:
            axis = 'y'

        return axis

    def get_logicals_x(self) -> Operator:
        """Get the unique logical X operator."""
        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, min(2*Lx, 2*Ly), 2):
            operator[(x, 2*Ly - x, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> Operator:
        """Get the unique logical Z operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for z in range(1, 2*Lz, 2):
            for x in range(1, min(2*Lx, 2*Ly), 2):
                operator[(x, x, z)] = 'Z'
        logicals.append(operator)

        return logicals

    def qubit_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().qubit_representation(location, rotated_picture)

        if self.qubit_axis(location) == 'z':
            representation['params']['length'] = 2

        return representation

    def stabilizer_representation(self, location, rotated_picture=False) -> Dict:
        representation = super().stabilizer_representation(location, rotated_picture)

        x, y, z = location
        if not rotated_picture and self.stabilizer_type(location) == 'face':
            if z % 2 == 1:
                representation['params']['normal'] = [0, 0, 1]
                representation['params']['angle'] = np.pi/4
            else:
                representation['params']['w'] = 1.5
                representation['params']['angle'] = 0

                if (x + y) % 4 == 0:
                    representation['params']['normal'] = [1, 1, 0]
                else:
                    representation['params']['normal'] = [-1, 1, 0]

        if rotated_picture and self.stabilizer_type(location) == 'face':
            if z % 2 == 1:
                representation['params']['normal'] = [0, 0, 1]
                representation['params']['angle'] = 0
            else:
                representation['params']['w'] = 1.4142
                representation['params']['h'] = 1.4142
                representation['params']['angle'] = np.pi/4

                if (x + y) % 4 == 0:
                    representation['params']['normal'] = [1, 1, 0]
                else:
                    representation['params']['normal'] = [-1, 1, 0]

        return representation
