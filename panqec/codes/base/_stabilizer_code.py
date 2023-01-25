from typing import Dict, Tuple, Optional, List
from abc import abstractmethod
import os
from ._subsystem_code import SubsystemCode

import panqec

os.environ['PANQEC_ROOT_DIR'] = os.path.dirname(panqec.__file__)

Operator = Dict[Tuple, str]  # Coordinate to pauli ('X', 'Y' or 'Z')


class StabilizerCode(SubsystemCode):
    """Abstract class for generic stabilizer codes (CSS or not)

    Any subclass should override the following four methods:
    - get_qubit_coordinates() to define all the coordinates in the lattice
    that contain qubits
    - get_stabilizer_coordinates() to define all the coordinates in the lattice
    that contain stabilizers
    - qubit_axis(location) to return the axis of a qubit at a given location
    (when qubit have an orientation in space, for instance when they are edges)

    Using only those methods, a StabilizerCode will then automatically create
    the corresponding parity-check matrix (in self.stabilizers) and can be used
    to make a visualization in the GUI or calculate thresholds.
    """

    def __init__(
        self,
        L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
    ):
        """Constructor for the StabilizerCode class

        Parameters
        ----------
        L_x : int
            Dimension of the lattice in the x direction (or in all directions
            if L_y and L_z are not given)
        L_y: int, optional
            Dimension of the lattice in the y direction
        L_z: int, optional
            Dimension of the lattice in the z direction
        """

        super().__init__(L_x, L_y, L_z)

    def get_gauge_coordinates(self) -> List[Tuple]:
        """"Coordinates of all the gauge operators.
        For stabilizer codes, those are the same as the stabilizers"""

        return self.get_stabilizer_coordinates()

    def get_gauge_operator(self, gauge_coordinate: Tuple) -> Operator:
        return self.get_stabilizer(gauge_coordinate)

    def gauge_type(self, gauge_coordinate: Tuple) -> str:
        return self.stabilizer_type(gauge_coordinate)

    def gauge_representation(self, gauge_coordinate: Tuple) -> Dict:
        return self.stabilizer_representation(gauge_coordinate)

    def get_stabilizer_gauge_operators(self, location: Tuple) -> Operator:
        return [location]

    @abstractmethod
    def get_stabilizer(self, location: Tuple) -> Operator:
        """ Returns a stabilizer, formatted as dictionary that assigns a Pauli
        operator ('X', 'Y' or 'Z') to each qubit location in the support of
        the stabilizer.

        For example, for a vertex stabilizer in the 2D toric code, we could
        have
        `get_stabilizer((1,1)) -> {(1,0): 'X', (0, 1): 'X', (2, 1): 'X',
        (1, 2): 'X'}`

        Parameters
        ----------
        location: Tuple
            Location of the stabilizer in the coordinate system

        Returns
        -------
        stabilizer: Dict[Tuple, str]
            Dictionary that assigns a Pauli operator ('X', 'Y' or 'Z') to each
            qubit location in the support of the stabilizer
        """
