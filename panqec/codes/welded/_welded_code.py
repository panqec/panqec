from typing import Tuple, Dict, List, Optional
import numpy as np
from functools import reduce
from itertools import product

from panqec.codes import StabilizerCode, Planar3DCode


Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class WeldedCode(StabilizerCode):
    """
    Parameters
    ----------
    L_x : int
        The size of the macro lattice in the x direction.
    L_y : Optional[int]
        The size of the macro lattice in the y direction.
        If it is not given, it is assumed to be a square lattice with Lx=Ly
    L_z : Optional[int]
        The size of the macro lattice in the z direction
        If it is not given, it is assumed to be a cubic lattice with Lx=Ly=Lz.
    base_code_class: type[StabilizerCode]
        The class of the base code to be welded (e.g. Toric3DCode)
    base_lattice_size: Tuple[int, int, int]
        Tuple of three integers corresponding to the lattice size of the base code
        (e.g. (4, 4, 4))
    x_welds:
    """
    dimension = 3
    deformation_names = []

    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
        base_code_class: type[StabilizerCode] = Planar3DCode,
        base_lattice_size: Tuple[int, int, int] = (4, 4, 4)
    ):
        self.base_code_class = base_code_class
        self.base_code = self.base_code_class(*base_lattice_size)
        self.base_lattice_size = base_lattice_size

        self.max_qubit_coordinates = [
            reduce(
                lambda a, b: max(a, b[i]),
                self.base_code.qubit_coordinates,
                0
            )
            for i in range(3)
        ]
        self.min_qubit_coordinates = [
            reduce(
                lambda a, b: min(a, b[i]),
                self.base_code.qubit_coordinates,
                np.inf
            )
            for i in range(3)
        ]

        super().__init__(L_x, L_y, L_z)

    @property
    def label(self) -> str:
        return 'Welded [{}] {}x{}x{}'.format(self.base_code.label, *self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly, Lz = self.size
        bLx, bLy, bLz = self.base_lattice_size

        def keep_qubit(coordinates):
            x, y, z, bx, by, bz = coordinates
            rough_x = (z % 2 == 1 and y % 2 == 1
                       and (bx == self.max_qubit_coordinates[0]
                            or bx == self.min_qubit_coordinates[0]))
            rough_y = (z % 2 == 1 and x % 2 == 1
                       and (by == self.max_qubit_coordinates[1]
                            or by == self.min_qubit_coordinates[1]))
            bottom_rough_z = (z % 2 == 1 and x % 2 == 1
                              and bz == self.min_qubit_coordinates[2])

            return not (rough_x or rough_y or bottom_rough_z)

        coordinates = []
        for x, y, z in product(range(Lx), range(Ly), range(Lz)):
            if (x + y + z) % 2 == 1:
                coordinates.extend(filter(keep_qubit, map(
                    lambda loc: (x, y, z, *loc),
                    self.base_code.qubit_coordinates)
                ))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        # Lx, Ly = self.size

        # for x, y, z in product(range(Lx), range(Ly), range(Lz)):
        #     if (x + y + z) % 2 == 1:
        #         coordinates.extend(filter(map(
        #             lambda loc: (x, y, z, *loc),
        #             self.base_code.qubit_coordinates),
        #             keep_qubit
        #         ))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        x, y = location
        if x % 2 == 0:
            return 'vertex'
        else:
            return 'face'

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        if self.stabilizer_type(location) == 'vertex':
            pauli = 'Z'
        else:
            pauli = 'X'

        delta: List[Tuple] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        operator: Operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d))

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location):
        x, y, z, bx, by, bz = location

        if (bz % 2 == 0) and (bx % 2 == 1) and (by % 2 == 0):
            axis = 'x'
        elif (bz % 2 == 0) and (bx % 2 == 0) and (by % 2 == 1):
            axis = 'y'
        elif (bz % 2 == 1) and (bx % 2 == 0) and (by % 2 == 0):
            axis = 'z'
        else:
            raise ValueError(f"Location {location} does not correspond"
                             "to a qubit")

        return axis

    def get_logicals_x(self) -> List[Operator]:
        # Lx, Ly = self.size
        logicals: List[Operator] = []

    #     # X operators along x edges in x direction.
    #     operator: Operator = dict()
    #     for x in range(1, 2*Lx, 2):
    #         operator[(x, 0)] = 'X'
    #     logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        # Lx, Ly = self.size
        logicals: List[Operator] = []

        # # X operators along x edges in x direction.
        # operator: Operator = dict()
        # for y in range(0, 2*Ly, 2):
        #     operator[(1, y)] = 'Z'
        # logicals.append(operator)

        return logicals

    def qubit_representation(
        self, location, rotated_picture=False, json_file=None
    ) -> Dict:
        Lx, Ly, Lz = self.size

        rep = super().qubit_representation(
            location, rotated_picture, json_file
        )

        x, y, z, bx, by, bz = location
        rep['location'] = (
            bx + x * (2*Lx+1), by + y * (2*Ly+1), bz + z * (2*Lz+1)
        )

        return rep
