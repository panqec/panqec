from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations


class RotatedPlanar2DCode(StabilizerCode):
    dimension = 2
    deformation_names = ['XZZX', 'XY']

    @property
    def label(self) -> str:
        return 'Rotated Planar {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly+1, 2):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Vertices
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly+1, 2):
                if (x + y) % 4 == 2:
                    coordinates.append((x, y))

        # Faces
        for x in range(0, 2*Lx+1, 2):
            for y in range(2, 2*Ly, 2):
                if (x + y) % 4 == 0:
                    coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location: Tuple):
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y = location
        if (x + y) % 4 == 2:
            return 'vertex'
        else:
            return 'face'

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        if self.stabilizer_type(location) == 'vertex':
            pauli = 'Z'
        else:
            pauli = 'X'

        delta = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location):
        x, y = location

        if (x + y) % 4 == 2:
            axis = 'x'
        elif (x + y) % 4 == 0:
            axis = 'y'
        else:
            raise ValueError(f'Location {location} does not correspond'
                             f'to a qubit')

        return axis

    def get_logicals_x(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals = []

        # X operators along first diagonal
        operator: Operator = dict()
        for x in range(1, 2*Lx+1, 2):
            operator[(x, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        Lx, Ly = self.size
        logicals = []

        # Z operators along first diagonal
        operator: Operator = dict()
        for y in range(1, 2*Ly+1, 2):
            operator[(1, y)] = 'Z'
        logicals.append(operator)

        return logicals

    def get_deformation(
        self, location: Tuple,
        deformation_name: str,
        deformation_axis: str = 'y',
        **kwargs
    ) -> Dict:

        if deformation_axis not in ['x', 'y']:
            raise ValueError(f"{deformation_axis} is not a valid "
                             "deformation axis")

        if deformation_name == 'XZZX':
            undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

            if self.qubit_axis(location) == deformation_axis:
                deformation = deformed_dict
            else:
                deformation = undeformed_dict

        elif deformation_name == 'XY':
            deformation = {'X': 'X', 'Y': 'Z', 'Z': 'Y'}

        else:
            raise ValueError(f"The deformation {deformation_name}"
                             "does not exist")

        return deformation

    def stabilizer_representation(
        self,
        location: Tuple,
        rotated_picture=False,
        json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(
            location, rotated_picture, json_file
        )

        Lx, Ly = self.size
        x, y = location

        if rotated_picture:
            if x == 0 or x == 2*Lx or y == 0 or y == 2*Ly:
                rep['object'] = 'semicircle'

                rep['params'] = {
                    'radius': 1,
                    'normal': [0, 0, 1],
                    'angle': 0
                }

                if x == 0:
                    rep['location'] = (x + 1, y)
                    rep['params']['angle'] = np.pi / 2

                if y == 0:
                    rep['location'] = (x, y + 1)
                    rep['params']['angle'] = np.pi

                if x == 2*Lx:
                    rep['location'] = (x - 1, y)
                    rep['params']['angle'] = - np.pi / 2

                if y == 2*Ly:
                    rep['location'] = (x, y - 1)

        return rep
