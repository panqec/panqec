from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class RotatedPlanar3DCode(StabilizerCode):
    """3D surface code with open boundaries on lattice rotated about z axis.

    Uses roughly half as many qubits as
    :class:`panqec.codes.surface_3d.Planar3DCode`.

    Parameters
    ----------
    L_x : int
        Number of qubits in the x direction.
    L_y : Optional[int
        Number of qubits in the y direction.
    L_z : Optional[int]
        Number of qubits in the z direction.

    Notes
    -----
    The lattice is stacked with lattices like those in
    :class:`panqec.codes.surface_2d.RotatedPlanar2DCode`
    glued with vertical qubits in between each layer.
    """
    dimension = 3
    deformation_names = ['XZZX']

    @property
    def label(self) -> str:
        return 'Rotated Planar {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        Lx, Ly, Lz = self.size

        coordinates: Coordinates = []

        # Horizontal
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly + 1, 2):
                for z in range(2, 2*Lz, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))
        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
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

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y, z = location
        if (x + y) % 4 == 2 and z % 2 == 1:
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

        x, y, z = location

        if self.stabilizer_type(location) == 'vertex':
            delta = [
                (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0), (0, 0, -1),
                (0, 0, 1)
            ]
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
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        if location not in self.qubit_coordinates:
            raise ValueError(
                f'Location {location} does not correspond to a qubit'
            )

        if (z % 2 == 0):
            axis = 'z'
        elif (x + y) % 4 == 2:
            axis = 'x'
        elif (x + y) % 4 == 0:
            axis = 'y'

        return axis

    def get_logicals_x(self) -> List[Operator]:
        """Get the unique logical X operator."""
        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator: Operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 1, 1)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """Get the unique logical Z operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Z operators along x edges in x direction.
        operator: Operator = dict()

        for z in range(1, 2*Lz, 2):
            for y in range(1, 2*Ly, 2):
                operator[(1, y, z)] = 'Z'
        logicals.append(operator)

        return logicals

    def qubit_representation(
        self, location, rotated_picture=False, json_file=None
    ) -> Dict:
        representation = super().qubit_representation(
            location, rotated_picture, json_file
        )

        if self.qubit_axis(location) == 'z':
            representation['params']['length'] = 2

        if rotated_picture:
            x, y, z = representation['location']
            representation['location'] = (x, y, z*1.4142)

        return representation

    def stabilizer_representation(
        self, location, rotated_picture=False, json_file=None
    ) -> Dict:
        representation = super().stabilizer_representation(
            location, rotated_picture, json_file
        )

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
                representation['params']['angle'] = np.pi/4

                if (x + y) % 4 == 0:
                    representation['params']['normal'] = [1, 1, 0]
                else:
                    representation['params']['normal'] = [-1, 1, 0]

        if rotated_picture:
            x, y, z = representation['location']
            representation['location'] = (x, y, z*1.4142)

        return representation

    def get_deformation(
        self, location: Tuple,
        deformation_name: str,
        deformation_axis: str = 'z',
        **kwargs
    ) -> Dict:

        if deformation_axis not in ['x', 'y', 'z']:
            raise ValueError(f"{deformation_axis} is not a valid "
                             "deformation axis")

        undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}

        if deformation_name == 'XZZX':
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

        else:
            raise ValueError(f"The deformation {deformation_name}"
                             "does not exist")

        if self.qubit_axis(location) == deformation_axis:
            return deformed_dict
        else:
            return undeformed_dict
