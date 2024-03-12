from copy import deepcopy
import itertools
import numpy as np
from panqec.codes import StabilizerCode

Operator = dict[tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = list[tuple]  # list of locations


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
        size = [2*Lx, 2*Ly, 2*Lz]

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
                    if (x_cube + y_cube + z_cube) % 4 == 3:
                        tetrahedra = even_tetrahedra
                    else:
                        tetrahedra = odd_tetrahedra

                    cube_coord = [x_cube, y_cube, z_cube]

                    for tetrahedron in tetrahedra:
                        vertices = np.add(cube_coord, tetrahedron, dtype=np.int32)
                        tetra = Tetra(vertices.tolist())

                        coordinates.append(
                            Tetra(vertices.tolist())
                        )
                        for v in vertices:
                            v = np.mod(v, [2*Lx, 2*Ly, 2*Lz]).tolist()
                            if tuple(v) in self.vertex_to_tetras:
                                self.vertex_to_tetras[tuple(v)].append(tetra)
                            else:
                                self.vertex_to_tetras[tuple(v)] = [tetra]

                        for v1, v2 in itertools.combinations(vertices, 2):
                            new_v1 = deepcopy(v1)
                            new_v2 = deepcopy(v2)
                            for axis in range(3):
                                if v1[axis] == v2[axis] == size[axis]:
                                    new_v1[axis] = 0
                                    new_v2[axis] = 0

                            e = tuple(sorted([tuple(new_v1.tolist()), tuple(new_v2.tolist())]))

                            if e in self.edge_to_tetras.keys():
                                self.edge_to_tetras[e].append(tetra)
                            else:
                                self.edge_to_tetras[e] = [tetra]

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates = (
            list(self.vertex_to_tetras.keys())
            + list(self.edge_to_tetras.keys())
        )

        return coordinates

    def stabilizer_type(self, location: tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} "
                             "for a stabilizer")

        if len(location) == 2:
            return 'edge'
        else:
            return 'vertex'

    def get_stabilizer(self, location: tuple) -> Operator:
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

    def qubit_axis(self, location: Tetra):
        return 'x'

    def get_logicals_z(self) -> list[Operator]:
        """The 3 logical Z operators (errors on all squares
        of a given layer)"""

        Lx, Ly, Lz = self.size

        logicals = []

        return logicals

    def get_logicals_x(self) -> list[Operator]:
        """Get the 3 logical X operators."""
        Lx, Ly, Lz = self.size
        logicals = []

        return logicals

    def qubit_representation(
        self, location: Tetra, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().qubit_representation(location, rotated_picture)

        v = location.vertices
        vertices = (*v[0], *v[1], *v[2], *v[3])
        triangles = np.ravel(list(itertools.combinations(range(4), 3))).tolist()

        rep['params'] = {'vertices': vertices, 'triangles': triangles}
        rep['location'] = np.mean(v, axis=0, dtype=float).tolist()

        return rep

    def stabilizer_representation(
        self, location: tuple, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().stabilizer_representation(location, rotated_picture)
        Lx, Ly, Lz = self.size
        size = [2*Lx, 2*Ly, 2*Lz]

        if self.stabilizer_type(location) == 'edge':
            direction = [
                location[0][i] - location[1][i]
                for i in range(3)
            ]
            length = float(np.linalg.norm(direction))

            rep['params']['direction'] = direction
            rep['params']['length'] = length
            rep['location'] = np.mean(location, axis=0).tolist()

            if 0 in rep['location']:
                rep['object'] = 'group'
                edge_params = {
                    'object': 'cylinder', 'location': rep['location'], 'params': rep['params']
                }
                rep['params'] = [edge_params]

                for axis in range(3):
                    if rep['location'][axis] == 0:
                        n_new = len(rep['params'])
                        rep['params'].extend(deepcopy(rep['params']))

                        for i in range(n_new):
                            rep['params'][-1-i]['location'][axis] = size[axis]

        else:
            if 0 in location:
                rep['object'] = 'group'
                sphere_params = {
                    'object': 'sphere', 'location': list(location), 'params': rep['params']
                }
                rep['params'] = [sphere_params]

                for axis in range(3):
                    if location[axis] == 0:
                        n_new = len(rep['params'])
                        rep['params'].extend(deepcopy(rep['params']))
                        for i in range(n_new):
                            rep['params'][-1-i]['location'][axis] = size[axis]

        return rep
