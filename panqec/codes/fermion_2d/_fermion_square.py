from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class FermionSquare(StabilizerCode):
     """Compact fermion encoding, square lattice, periodic boundary.
    
    Parameters
    ----------
    L : int
        Number of fermions encoded. This equals to the 
        number of vertex qubits.

    Notes
    -----
    For more information: <https://arxiv.org/abs/2003.06939>
    """
    dimension = 2

    def __init__(self, L: int):
        super().__init__(L_x, None, None)
        self._qsmap = {}
        self._face_q_ids = []
        
    @property
    def label(self) -> str:
        return 'Fermionic Square {}x{}'.format(*self.size)


    def face_q_ids(self):
        if len(self._face_q_ids) == 0:
            for i, coord in enumerate(self.qubit_coordinates):
                if self.qubit_type(coord) == 'face':
                    self._face_q_ids.append(i)
        return self._face_q_ids
    
    def qsmap(self) -> Dict[int, List]:
        if len(self._qsmap.keys()) == 0:
            for qubit_location in self.qubit_index.keys():
                q_ind = self.qubit_index[qubit_location]
                self._qsmap[q_ind] = []
                if self.qubit_type(qubit_location) == 'face':
                    delta = [(0, 2), (0, -2), (2, 0), (-2, 0)] #first 2 - X error, last 2 - Y error
                else:
                    delta = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
                for d in delta:
                    stab_location = tuple(np.add(qubit_location, d) %
                                       (2*np.array(self.size)))
                    if self.is_stabilizer(stab_location):
                        self._qsmap[q_ind].append(self.stabilizer_index[stab_location])
        return self._qsmap
        
    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size
        
        # vertex qubits 
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                coordinates.append((x, y))

        # Face Qubits 
        for x in range(1, 2*Lx, 4):
            for y in range(1, 2*Ly, 4):
                coordinates.append((x, y))
 
        for x in range(3, 2*Lx, 4):
            for y in range(3, 2*Ly, 4):
                coordinates.append((x, y))
        
        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Faces
        for x in range(3, 2*Lx, 4):
            for y in range(1, 2*Ly, 4):
                coordinates.append((x, y))
                
        for x in range(1, 2*Lx, 4):
            for y in range(3, 2*Ly, 4):
                coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")
        
        return 'face'

    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")
        
        def get_pauli(relative_location):
                if relative_location == (0,2) or relative_location == (0,-2):
                    return 'Y'
                if relative_location == (2,0) or relative_location == (-2, 0):
                    return 'X'
                else:
                    return 'Z'

        delta = [(-1, 1), (1, 1), (1, -1), (-1, -1), (0, 2), (2, 0), (-2, 0), (0, -2)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d) %
                                   (2*np.array(self.size)))
            if self.is_qubit(qubit_location):
                operator[qubit_location] = (get_pauli(d))
                
        return operator

    def qubit_type(self, location) -> str:
        if location[0] % 2 != 0 and location[1] % 2 != 0:
            return 'face'
        else:
            return 'vertex'
    
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
        representation = super().stabilizer_representation(location, rotated_picture, json_file='FermionSquare.json')
        return representation

    def qubit_representation(self, location, rotated_picture=False):
        representation = super().qubit_representation(location, rotated_picture, json_file='FermionSquare.json')
        return representation