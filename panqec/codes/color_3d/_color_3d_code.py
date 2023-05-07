from typing import Tuple, Dict, List
import itertools
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations


class Color3DCode(StabilizerCode):
    """3D Color Code on periodic truncated octahedral lattice.

    Parameters
    ----------
    L_x : int
        Size in the x direction.
    L_y : int
        Size in the y direction.
    L_z : int
        Size in the z direction.

    Notes
    -----
    See
    `Bombin and Martin-Delgado 2007 <https://arxiv.org/abs/cond-mat/0607736>`_
    for more information.
    """
    dimension = 3

    @property
    def label(self) -> str:
        return 'Color {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        coordinates = []

        stab_coordinates = self.get_stabilizer_coordinates()

        for location in stab_coordinates:
            qubit_coords = list(self.get_stabilizer(location).keys())
            for coord in qubit_coords:
                if coord not in coordinates:
                    coordinates.append(coord)

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size

        # Yellow and red cells
        for z in range(2, 4*Lz, 4):
            for x in range(2, 4*Lx, 4):
                for y in range(2, 4*Ly, 4):
                    coordinates.append((x, y, z))

        # Blue and green cells
        for z in range(4, 4*Lz+1, 4):
            for x in range(4, 4*Lx+1, 4):
                for y in range(4, 4*Ly+1, 4):
                    coordinates.append((x, y, z))

        # Yellow and red square faces orthogonal to z axis
        for z in range(0, 4*Lz, 4):
            for x in range(2, 4*Lx, 4):
                for y in range(2, 4*Ly, 4):
                    coordinates.append((x, y, z))

        # Yellow and red square faces orthogonal to x axis
        for z in range(2, 4*Lz, 4):
            for x in range(0, 4*Lx, 4):
                for y in range(2, 4*Ly, 4):
                    coordinates.append((x, y, z))

        # Yellow and red square faces orthogonal to y axis
        for z in range(2, 4*Lz, 4):
            for x in range(2, 4*Lx, 4):
                for y in range(0, 4*Ly, 4):
                    coordinates.append((x, y, z))

        # Blue and green square faces orthogonal to z axis
        for z in range(2, 4*Lz+1, 4):
            for x in range(4, 4*Lx+1, 4):
                for y in range(4, 4*Ly+1, 4):
                    coordinates.append((x, y, z))

        # Blue and green square faces orthogonal to x axis
        for z in range(4, 4*Lz+1, 4):
            for x in range(2, 4*Lx+1, 4):
                for y in range(4, 4*Ly+1, 4):
                    coordinates.append((x, y, z))

        # Blue and green square faces orthogonal to y axis
        for z in range(4, 4*Lz+1, 4):
            for x in range(4, 4*Lx+1, 4):
                for y in range(2, 4*Ly+1, 4):
                    coordinates.append((x, y, z))

        # All hexagons
        for z in range(1, 4*Lz+2, 2):
            for x in range(1, 4*Lx+2, 2):
                for y in range(1, 4*Ly+2, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        x, y, z = location

        cell_color = {6: 'cell-yellow',
                      2: 'cell-red',
                      4: 'cell-green',
                      0: 'cell-blue'}

        if x % 2 == 1:
            stab_type = 'face-hex'
        elif x % 4 == z % 4 and y % 4 == z % 4:
            stab_type = cell_color[(x + y + z) % 8]
        else:
            stab_type = 'face-square'

        return stab_type

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        Lx, Ly, Lz = self.size

        if 'cell' in self.stabilizer_type(location):
            pauli = 'Z'
        else:
            pauli = 'X'

        x, y, z = location

        if 'cell' in self.stabilizer_type(location):
            # Cell coordinates consist of all permutations of {0,1,2} with
            # all possible signs in front of 1 and 2
            signs = np.array(list(itertools.product([-1, 1], [-1, 1])))
            permutations = np.vstack([signs * [1, 2], signs * [2, 1]])
            delta = np.vstack([np.insert(permutations, 0, 0, axis=1),
                               np.insert(permutations, 1, 0, axis=1),
                               np.insert(permutations, 2, 0, axis=1)]
                              ).tolist()

        elif self.stabilizer_type(location) == 'face-square':
            if x % 4 == z % 4:  # xz square
                delta = [(0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0)]
            elif y % 4 == z % 4:  # yz square
                delta = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)]
            else:  # xy square
                delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]

        elif self.stabilizer_type(location) == 'face-hex':
            if x % 4 == z % 4 and y % 4 == z % 4:
                # print("Hex 1")
                delta = [(-1, 0, 1), (1, 0, -1),
                         (0, -1, 1), (0, 1, -1),
                         (-1, 1, 0), (1, -1, 0)]
            elif x % 4 != z % 4 and y % 4 == z % 4:
                # print("Hex 2")
                delta = [(-1, 0, -1), (1, 0, 1),
                         (0, -1, 1), (0, 1, -1),
                         (-1, -1, 0), (1, 1, 0)]
            elif x % 4 == z % 4 and y % 4 != z % 4:
                # print("Hex 3")
                delta = [(-1, 0, 1), (1, 0, -1),
                         (0, -1, -1), (0, 1, 1),
                         (-1, -1, 0), (1, 1, 0)]
            else:
                # print("Hex 4")
                delta = [(-1, 0, -1), (1, 0, 1),
                         (0, -1, -1), (0, 1, 1),
                         (-1, 1, 0), (1, -1, 0)]

        operator: Operator = dict()
        for d in delta:
            qubit_location = (int(x + d[0]) % (4 * Lx),
                              int(y + d[1]) % (4 * Ly),
                              int(z + d[2]) % (4 * Lz))

            operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        return 'x'

    def get_logicals_z(self) -> List[Operator]:
        """The 3 logical Z operators (errors on all squares
        of a given layer)"""

        Lx, Ly, Lz = self.size

        logicals = []

        # Yellow-red membrane yz plane
        x = 0
        operator: Operator = dict()
        for y in range(2, 4*Ly, 4):
            for z in range(2, 4*Lz, 4):
                operator[(x, (y + 1) % (4 * Ly), z)] = 'Z'
                operator[(x, (y - 1) % (4 * Ly), z)] = 'Z'
                operator[(x, y, (z + 1) % (4 * Lz))] = 'Z'
                operator[(x, y, (z - 1) % (4 * Lz))] = 'Z'
        logicals.append(operator)

        # Green-blue membrane yz plane
        x = 2
        operator = dict()
        for y in range(0, 4*Ly, 4):
            for z in range(0, 4*Lz, 4):
                operator[(x, (y + 1) % (4 * Ly), z)] = 'Z'
                operator[(x, (y - 1) % (4 * Ly), z)] = 'Z'
                operator[(x, y, (z + 1) % (4 * Lz))] = 'Z'
                operator[(x, y, (z - 1) % (4 * Lz))] = 'Z'
        logicals.append(operator)

        # Red-green membrane yz plane
        x = 3
        operator = dict()
        for y in range(1, 4*Ly+2, 2):
            for z in range(1, 4*Lz+2, 2):
                cell_loc_1 = ((x-1) % (4*Lx), (y-1) % (4 * Ly), (z+1) % (4*Lz))
                cell_loc_2 = ((x-1) % (4*Lx), (y+1) % (4 * Ly), (z-1) % (4*Lz))
                if ((self.is_stabilizer(cell_loc_1) and
                        self.stabilizer_type(cell_loc_1) == 'cell-red') or
                        (self.is_stabilizer(cell_loc_2) and
                         self.stabilizer_type(cell_loc_2) == 'cell-red')):
                    for qubit_loc in self.get_stabilizer((x, y, z)).keys():
                        operator[qubit_loc] = 'Z'
        logicals.append(operator)

        # Yellow-red membrane xz plane
        y = 0
        operator = dict()
        for x in range(2, 4*Lx, 4):
            for z in range(2, 4*Lz, 4):
                operator[((x + 1) % (4 * Lx), y, z)] = 'Z'
                operator[((x - 1) % (4 * Lx), y, z)] = 'Z'
                operator[(x, y, (z + 1) % (4 * Lz))] = 'Z'
                operator[(x, y, (z - 1) % (4 * Lz))] = 'Z'
        logicals.append(operator)

        # Green-blue membrane xz plane
        y = 2
        operator = dict()
        for x in range(0, 4*Lx, 4):
            for z in range(0, 4*Lz, 4):
                operator[((x + 1) % (4*Lx), y, z)] = 'Z'
                operator[((x - 1) % (4*Lx), y, z)] = 'Z'
                operator[(x, y, (z + 1) % (4 * Lz))] = 'Z'
                operator[(x, y, (z - 1) % (4 * Lz))] = 'Z'
        logicals.append(operator)

        # Red and green membrane xz plane
        y = 3
        operator = dict()
        for x in range(1, 4*Lx+2, 2):
            for z in range(1, 4*Lz+2, 2):
                cell_loc_1 = ((x-1) % (4*Lx), (y-1) % (4 * Ly), (z+1) % (4*Lz))
                cell_loc_2 = ((x+1) % (4*Lx), (y-1) % (4 * Ly), (z-1) % (4*Lz))
                if ((self.is_stabilizer(cell_loc_1) and
                        self.stabilizer_type(cell_loc_1) == 'cell-red') or
                        (self.is_stabilizer(cell_loc_2) and
                         self.stabilizer_type(cell_loc_2) == 'cell-red')):
                    for qubit_loc in self.get_stabilizer((x, y, z)).keys():
                        operator[qubit_loc] = 'Z'
        logicals.append(operator)

        # Yellow-red membrane xy plane
        z = 0
        operator = dict()
        for x in range(2, 4*Lx, 4):
            for y in range(2, 4*Ly, 4):
                operator[((x + 1) % (4 * Lx), y, z)] = 'Z'
                operator[((x - 1) % (4 * Lx), y, z)] = 'Z'
                operator[(x, (y + 1) % (4 * Ly), z)] = 'Z'
                operator[(x, (y - 1) % (4 * Ly), z)] = 'Z'
        logicals.append(operator)

        # Green-blue membrane xy plane
        z = 2
        operator = dict()
        for x in range(0, 4*Lx, 4):
            for y in range(0, 4*Ly, 4):
                operator[((x + 1) % (4 * Lx), y, z)] = 'Z'
                operator[((x - 1) % (4 * Lx), y, z)] = 'Z'
                operator[(x, (y + 1) % (4 * Ly), z)] = 'Z'
                operator[(x, (y - 1) % (4 * Ly), z)] = 'Z'
        logicals.append(operator)

        # Red and green membrane xy plane
        z = 3
        operator = dict()
        for x in range(1, 4*Lx+2, 2):
            for y in range(1, 4*Ly+2, 2):
                cell_loc_1 = ((x-1) % (4*Lx), (y+1) % (4 * Ly), (z-1) % (4*Lz))
                cell_loc_2 = ((x+1) % (4*Lx), (y-1) % (4 * Ly), (z-1) % (4*Lz))
                if ((self.is_stabilizer(cell_loc_1) and
                        self.stabilizer_type(cell_loc_1) == 'cell-red') or
                        (self.is_stabilizer(cell_loc_2) and
                         self.stabilizer_type(cell_loc_2) == 'cell-red')):
                    for qubit_loc in self.get_stabilizer((x, y, z)).keys():
                        operator[qubit_loc] = 'Z'
        logicals.append(operator)

        return logicals

    def get_logicals_x(self) -> List[Operator]:
        """Get the 3 logical X operators."""
        Lx, Ly, Lz = self.size
        logicals = []

        operator: Operator = dict()
        for x in range(2, 4*Lx-2, 8):
            operator[(x-2, 6, 1)] = 'X'
            operator[(x-1, 6, 0)] = 'X'
            operator[(x+1, 6, 0)] = 'X'
            operator[(x+2, 6, 1)] = 'X'
        logicals.append(operator)

        operator = dict()
        for x in range(4, 4*Lx-2, 8):
            operator[(x-2, 0, 1)] = 'X'
            operator[(x-1, 0, 2)] = 'X'
            operator[(x+1, 0, 2)] = 'X'
            operator[(x+2, 0, 1)] = 'X'
        logicals.append(operator)

        operator = dict()
        for x in range(1, 4*Lx, 2):
            operator[(x, 2, 0)] = 'X'
        logicals.append(operator)

        operator = dict()
        for y in range(2, 4*Ly-2, 8):
            operator[(6, y-2, 1)] = 'X'
            operator[(6, y-1, 0)] = 'X'
            operator[(6, y+1, 0)] = 'X'
            operator[(6, y+2, 1)] = 'X'
        logicals.append(operator)

        operator = dict()
        for y in range(4, 4*Ly-2, 8):
            operator[(0, y-2, 1)] = 'X'
            operator[(0, y-1, 2)] = 'X'
            operator[(0, y+1, 2)] = 'X'
            operator[(0, y+2, 1)] = 'X'
        logicals.append(operator)

        operator = dict()
        for y in range(1, 4*Ly, 2):
            operator[(2, y, 0)] = 'X'
        logicals.append(operator)

        operator = dict()
        for z in range(2, 4*Lz-2, 8):
            operator[(6, 1, z-2)] = 'X'
            operator[(6, 0, z-1)] = 'X'
            operator[(6, 0, z+1)] = 'X'
            operator[(6, 1, z+2)] = 'X'
        logicals.append(operator)

        operator = dict()
        for z in range(4, 4*Lz-2, 8):
            operator[(0, 1, z-2)] = 'X'
            operator[(0, 2, z-1)] = 'X'
            operator[(0, 2, z+1)] = 'X'
            operator[(0, 1, z+2)] = 'X'
        logicals.append(operator)

        operator = dict()
        for z in range(1, 4*Lz, 2):
            operator[(2, 0, z)] = 'X'
        logicals.append(operator)

        return logicals

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(location, rotated_picture)

        x, y, z = location

        if self.stabilizer_type(location) == 'face-square':
            if x % 4 == z % 4:
                rep['params']['normal'] = [0, 1, 0]
            elif y % 4 == z % 4:
                rep['params']['normal'] = [1, 0, 0]
            else:
                rep['params']['normal'] = [0, 0, 1]
        elif self.stabilizer_type(location) == 'face-hex':
            # No idea where the sqrt(2)/2 comes from, but it seems necessary
            if x % 4 == z % 4 and y % 4 == z % 4:
                rep['params']['normal'] = [1, 1, np.sqrt(2)/2]
            elif x % 4 != z % 4 and y % 4 == z % 4:
                rep['params']['normal'] = [1, -1, -np.sqrt(2)/2]
            elif x % 4 == z % 4 and y % 4 != z % 4:
                rep['params']['normal'] = [1, -1, np.sqrt(2)/2]
            else:
                rep['params']['normal'] = [1, 1, -np.sqrt(2)/2]

        return rep
