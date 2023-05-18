from typing import Tuple, Dict, List
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class Fermion3DCode(StabilizerCode):
    """Compact 3D fermion encoding, periodic boundary.
    
    Parameters
    ----------
    Lx : int
        dimension of the lattice in x direction 
        (only counting vertex qubits)
    Ly : int
        dimension of the lattice in y direction 
        (only counting vertex qubits)
    Lz : int
        dimension of the lattice in z direction 
        (only counting vertex qubits)

    
    Notes
    -----
    For more information: <https://arxiv.org/pdf/2101.10735.pdf>
    """

    dimension = 3

    @property
    def label(self):
        return 'Fermion 3D {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        # Vertex Qubits
        for x in range(0, 2*Lx-1, 2):
            for y in range(0, 2*Ly-1, 2):
                for z in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))

        # face qubits z-planes
        for x in range(1, 2*Lx-1, 4):
            for y in range(1, 2*Ly-1, 4):
                for z in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))
                    
        for x in range(3, 2*Lx-1, 4):
            for y in range(3, 2*Ly-1, 4):
                for z in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))
                    
        # face qubits x-planes
        for x in range(0, 2*Lz-1, 2):
            for y in range(1, 2*Ly-1, 4):
                for z in range(1, 2*Lx-1, 4):
                    coordinates.append((x, y, z))
                    
        for x in range(0, 2*Lz-1, 2):
            for y in range(3, 2*Ly-1, 4):
                for z in range(3, 2*Lx-1, 4):
                    coordinates.append((x, y, z))
                    
                    
        # face qubits y-planes
        for x in range(1, 2*Lx-1, 4):
            for y in range(0, 2*Lz-1, 2):
                for z in range(1, 2*Lx-1, 4):
                    coordinates.append((x, y, z))
                    
        for x in range(3, 2*Lx-1, 4):
            for y in range(0, 2*Lz-1, 2):
                for z in range(3, 2*Lx-1, 4):
                    coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self):
        coordinates = []
        Lx, Ly, Lz = self.size
        
        # Faces on z planes
        for x in range(3, 2*Lx-1, 4):
            for y in range(1, 2*Ly-1, 4):
                for z in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))
                    
        for x in range(1, 2*Lx-1, 4):
            for y in range(3, 2*Ly-1, 4):
                for z in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))
                    
        # Faces on x planes
        for x in range(0, 2*Lz-1, 2):
            for y in range(1, 2*Ly-1, 4):
                for z in range(3, 2*Lx-1, 4): 
                    coordinates.append((x, y, z))
                    
        for z in range(1, 2*Lx-1, 4):
            for y in range(3, 2*Ly-1, 4):
                for x in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))
        
        # Faces on y planes
        for y in range(0, 2*Lz-1, 2):
            for x in range(1, 2*Ly-1, 4):
                for z in range(3, 2*Lx-1, 4): 
                    coordinates.append((x, y, z))
                    
        for z in range(1, 2*Lx-1, 4):
            for x in range(3, 2*Ly-1, 4):
                for y in range(0, 2*Lz-1, 2):
                    coordinates.append((x, y, z))
        
        return coordinates

    def stabilizer_type(self, location):
            return 'face'

    def get_stabilizer(self, location, deformed_axis=None)-> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")
        
        x, y, z = location
        

        def get_pauli(relative_location, plane):
                if plane == 'z':
                    if relative_location == (0, 2, 0) or relative_location == (0, -2, 0):
                        return 'Y'
                    if relative_location == (2, 0, 0) or relative_location == (-2, 0, 0):
                        return 'X'
                    else:
                        return 'Z'
                if plane == 'x':
                    if relative_location == (0, 0, 2) or relative_location == (0, 0, -2):
                        return 'Y'
                    if relative_location == (0, 2, 0) or relative_location == (0, -2, 0):
                        return 'X'
                    else:
                        return 'Z'
                if plane == 'y':
                    if relative_location == (0, 0, 2) or relative_location == (0, 0, -2):
                        return 'Y'
                    if relative_location == (2, 0, 0) or relative_location == (-2, 0, 0):
                        return 'X'
                    else:
                        return 'Z'

        delta_z = [(-1, 1, 0), (1, 1, 0), (1, -1, 0), (-1, -1, 0), (0, 2, 0), (2, 0, 0), (-2, 0, 0), (0, -2, 0)]
        delta_x = [(0, -1, 1), (0, 1, 1), (0, 1, -1), (0, -1, -1), (0, 0, 2), (0, 2, 0), (0, -2, 0), (0, 0, -2)]
        delta_y = [(-1, 0, 1), (1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 0, 2), (2, 0, 0), (-2, 0, 0), (0, 0, -2)]
        
        if x % 2 == 0:
                plane = 'x'
                delta = delta_x
        if z % 2 == 0:
                plane = 'z'
                delta = delta_z
        if y % 2 == 0:
                plane = 'y'
                delta = delta_y
        
        operator = dict()
        
        for d in delta:
            qubit_location = tuple(np.add(location, d) %
                                   (2*np.array(self.size)))
            if self.is_qubit(qubit_location):
                operator[qubit_location] = (get_pauli(d, plane))
                
        return operator

    def get_logicals_x(self):
        Lx, Ly, Lz = self.size
        logicals = []


        return logicals

    def qubit_axis(self, location):
            return 'x'

    def get_logicals_z(self):
        Lx, Ly, Lz = self.size
        logicals = []

        return logicals

    def stabilizer_representation(self, location, rotated_picture=False):
        representation = super().stabilizer_representation(location, rotated_picture, json_file='toric3d.json')

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

    def qubit_representation(self, location, rotated_picture=False):
        representation = super().qubit_representation(location, rotated_picture, json_file='Fermion3d.json')

        return representation