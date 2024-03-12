from typing import Tuple, Dict, List
import itertools
import numpy as np
from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations


class Tetra:
    """Class to define the coordinates of a tetrahedron.
    We use the coordinates of the four vertices of the tetrahedron,
    and make a hash that is invariant with respect to vertex permutations
    """
    def __init__(self, vertices):
        self.vertices = sorted([tuple(v) for v in vertices])

    def __hash__(self):
        return hash((tuple(v) for v in self.vertices))

    def __eq__(self, other: "Tetra"):
        rep1 = tuple((tuple(v) for v in self.vertices))
        rep2 = tuple((tuple(v) for v in other.vertices))
        return rep1 == rep2

    def __repr__(self):
        return f"Tetra({self.vertices})"

    def __str__(self):
        return repr(self)


class GaugeColor3DCode(StabilizerCode):
    """3D color code on a lattice defined in Supp. Fig. 2 of arXiv:1503.08217

    Parameters
    ----------
    L_x : int
        Size in the x direction.
    L_y : int
        Size in the y direction.
    L_z : int
        Size in the z direction.
    """
    dimension = 3

    @property
    def label(self) -> str:
        return 'Gauge Color {}x{}x{}'.format(*self.size)

    def get_qubit_coordinates(self) -> Coordinates:
        Lx, Ly, Lz = self.size
        coordinates: Coordinates = []

        z_rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)

        even_tetrahedra = np.array([
            [[1, -1, -1], [-1, -1, -1], [1, 1, -1], [1, -1, 1]],
            [[-1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1]],
            [[-1, 1, 1], [1, -1, 1], [1, 1, 1], [1, 1, -1]],
            [[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]],
        ], dtype=np.int8)

        odd_tetrahedra = np.array([
            [z_rot_mat.dot(vertex) for vertex in tetra]
            for tetra in even_tetrahedra
        ], dtype=np.int8)

        self.vertex_to_tetras: dict[tuple, list] = dict()
        self.edge_to_tetras: dict[tuple, list] = dict()

        for x_cube in range(1, 2*Lx, 2):
            for y_cube in range(1, 2*Ly, 2):
                for z_cube in range(1, 2*Lz, 2):
                    if (x_cube + y_cube + z_cube) % 4 == 0:
                        tetrahedra = even_tetrahedra
                    else:
                        tetrahedra = odd_tetrahedra

                    cube_coord = [x_cube, y_cube, z_cube]

                    for tetrahedron in tetrahedra:
                        vertices = np.add(cube_coord, tetrahedron, dtype=np.int32)
                        tetra = Tetra(vertices)

                        coordinates.append(
                            Tetra(vertices)
                        )
                        for v in vertices:
                            v = np.mod(v, [2*Lx, 2*Ly, 2*Lz])
                            if tuple(v) in self.vertex_to_tetras:
                                self.vertex_to_tetras[tuple(v)].append(tetra)
                            else:
                                self.vertex_to_tetras[tuple(v)] = [tetra]

                        for v1, v2 in itertools.combinations(vertices, 2):
                            e = (v1 + v2) // 2
                            e = np.mod(e, [2*Lx, 2*Ly, 2*Lz])

                            if tuple(e) in self.edge_to_tetras:
                                self.edge_to_tetras[tuple(e)].append(tetra)
                            else:
                                self.edge_to_tetras[tuple(e)] = [tetra]

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = (
            list(self.vertex_to_tetras.keys())
            + list(self.edge_to_tetras.keys())
        )

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        x, y, z = location

        if x % 2 == y % 2 == z % 2 == 0:
            return 'vertex'
        else:
            return 'edge'

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location}"
                             "for a stabilizer")

        operator: Operator = dict()
        if self.stabilizer_type(location) == 'vertex':
            pauli = 'X'
            for tetra in self.vertex_to_tetras[location]:
                operator[tetra] = pauli

        else:
            pauli = 'Z'
            for tetra in self.edge_to_tetras[location]:
                operator[tetra] = pauli

        return operator

    def qubit_axis(self, location):
        return 'x'

    def get_logicals_z(self) -> List[Operator]:
        """The 3 logical Z operators (errors on all squares
        of a given layer)"""

        Lx, Ly, Lz = self.size

        logicals = []

        return logicals

    def get_logicals_x(self) -> List[Operator]:
        """Get the 3 logical X operators."""
        Lx, Ly, Lz = self.size
        logicals = []

        return logicals

    def qubit_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(location, rotated_picture)

        return rep

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> Dict:
        rep = super().stabilizer_representation(location, rotated_picture)

        return rep