from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class Toric2DCode(StabilizerCode):
    """The original 2D toric code introduced by Kitaev.
    The qubits live on the edges of a 2D periodic square lattice.
    There are two types of stabilizer generators:
    vertex operators on vertices, and face operators faces.

    The coordinate system used is shown below.

    .. image:: toric_2d_code.svg
        :scale: 200 %
        :align: center

    Parameters
    ----------
    L_x : int
        The size in the x direction.
    L_y : Optional[int]
        The size in the y direction.
        If it is not given, it is assumed to be a square lattice with Lx=Ly.
    """
    dimension = 2
    deformation_names = ['XZZX', 'XY']

    @property
    def label(self) -> str:
        return 'Toric {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Qubits along e_y
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Vertices
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Faces
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        x, y = location
        if x % 2 == 0:
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

        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d) %
                                   (2*np.array(self.size)))

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location) -> str:
        x, y = location

        if (x % 2 == 1) and (y % 2 == 0):
            axis = 'x'
        elif (x % 2 == 0) and (y % 2 == 1):
            axis = 'y'
        else:
            raise ValueError(f'Location {location} does not correspond'
                             'to a qubit')

        return axis

    def get_logicals_x(self) -> List[Operator]:
        """The 2 logical X operators."""

        Lx, Ly = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator: Operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 0)] = 'X'
        logicals.append(operator)

        # X operators along y edges in y direction.
        operator = dict()
        for y in range(1, 2*Ly, 2):
            operator[(0, y)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """The 2 logical Z operators."""

        Lx, Ly = self.size
        logicals = []

        # Z operators on x edges forming surface normal to x (yz plane).
        operator: Operator = dict()
        for y in range(0, 2*Ly, 2):
            operator[(1, y)] = 'Z'
        logicals.append(operator)

        # Z operators on y edges forming surface normal to y (zx plane).
        operator = dict()
        for x in range(0, 2*Lx, 2):
            operator[(x, 1)] = 'Z'
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
