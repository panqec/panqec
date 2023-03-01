from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations


class RotatedPlanar2DCode(StabilizerCode):
    """Rotated 2D surface code with open boundaries.
    These use roughly half the number of qubits but only encode 1 logical
    qubit.

    Parameters
    ----------
    L_x : int
        Number of qubits in the x direction.
    L_y : Optional[int]
        Number of qubits in the y direction.
        Assumed square if not given.

    Notes
    -----
    One stabilizer of each type is shown in the figure below.
    In this picture, the qubits live on the edges and there
    are two types of stabilizers: vertex Z stabilizers and
    face X stabilizers.
    Note the weight-2 stabilizers on the boundaries.
    The below lattice is of size 5 by 4.

    .. image:: rotated_planar_2d_code.svg
        :scale: 200 %
        :align: center

    Alternatively, we can consider the the same coordinates,
    qubits and stabilizers but seen on a rotated lattice,
    where the lattice live on vertices,
    and the two types of stabilizers correspond to the color
    of the checkerboard faces as shown below.

    .. image:: rotated_planar_2d_code_2.svg
        :scale: 200 %
        :align: center
    """
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
