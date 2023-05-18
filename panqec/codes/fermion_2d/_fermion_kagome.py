from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class FermionKagome(StabilizerCode):
    """Compact fermion encoding, Kagome lattice (3.6.3.6 Uniform Tiling), periodic boundary.
    
    Parameters
    ----------
    Lx : int
        Number of columns of stabilizers.
    Ly : int
        Number of rows of stabilizers.

    Notes
    -----
    For more information: <https://arxiv.org/pdf/2101.10735.pdf>
    """

    dimension = 2

    @property
    def label(self) -> str:
        return 'Fermionic Kagome {}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # vertex qubits
        for x in range(1, 4*Lx, 2):
            for y in range(0, 4*Ly, 4):
                coordinates.append((x, y))
        
        for x in range(0, 4*Lx, 4):
            for y in range(2, 4*Ly, 8):
                coordinates.append((x, y))
                
        for x in range(2, 4*Lx, 4):
            for y in range(6, 4*Ly, 8):
                coordinates.append((x, y))
        
        # Face Qubits 
        for x in range(4, 4*Lx, 4):
            for y in range(1, 4*Ly, 8):
                coordinates.append((x, y))
                
        for x in range(4, 4*Lx, 4):
            for y in range(3, 4*Ly, 8):
                coordinates.append((x, y))

        for x in range(2, 4*Lx, 4):
            for y in range(5, 4*Ly, 8):
                coordinates.append((x, y))
                
        for x in range(2, 4*Lx, 4):
            for y in range(7, 4*Ly, 8):
                coordinates.append((x, y))
        
                
        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size
        for x in range(2, 4*Lx, 4):
            for y in range(2, 4*Ly, 8):
                coordinates.append((x, y))
        
        for x in range(4, 4*Lx, 4):
            for y in range(6, 4*Ly, 8):
                coordinates.append((x, y))
        
        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")
        return 'face'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")
        

        delta = {(2,0):'Z', (-2,0):'Z', (1,2):'Z' , (-1,2):'Z', (1,-2):'Z', (-1,-2):'Z', (0,3):'Z', (0,-3):'Z',
                (2,1):'X', (-2, -1): 'X',
                (-2, 1):'Y', (2,-1): 'Y'}

        operator = dict()

        for key in delta.keys():
            qubit_location = tuple(np.add(location, key) %
                                  (4*np.array(self.size)))
            if self.is_qubit(qubit_location):
                operator[qubit_location] = delta[key]
                
        return operator

    def qubit_axis(self, location) -> str:
        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        """The 2 logical X operators."""

        Lx, Ly = self.size
        logicals = []

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """The 2 logical Z operators."""

        Lx, Ly = self.size
        logicals = []

        return logicals
    
    def stabilizer_representation(self, location, rotated_picture=False):
        representation = super().stabilizer_representation(location, rotated_picture, json_file='FermionKagome.json')
        return representation

    def qubit_representation(self, location, rotated_picture=False):
        representation = super().qubit_representation(location, rotated_picture, json_file='FermionKagome.json')
        return representation