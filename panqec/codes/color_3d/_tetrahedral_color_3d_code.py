from __future__ import annotations

from collections import deque
from itertools import combinations
from math import atan2, sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np

from panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')
Coordinates = List[Tuple]  # List of locations
Node = Tuple
Edge = Tuple[Node, Node]


class Tetra:
    """Hashable dual tetrahedron coordinate.

    Qubits in the primal picture sit at the barycentres of these dual
    tetrahedra.
    """

    def __init__(self, vertices: Iterable[Node]):
        self.vertices = tuple(sorted(tuple(vertex) for vertex in vertices))

    def __hash__(self):
        return hash(self.vertices)

    def __eq__(self, other: object):
        if not isinstance(other, Tetra):
            return False

        return self.vertices == other.vertices

    def __lt__(self, other: "Tetra"):
        return self.vertices < other.vertices

    def __repr__(self):
        return f"Tetra({self.vertices})"

    def __str__(self):
        return repr(self)


class TetrahedralColor3DCode(StabilizerCode):
    """Tetrahedral 3D color code.

    The input ``L_x`` is interpreted as the odd code distance ``d``.  The
    implementation follows the dual tetrahedral complex convention: physical
    qubits are dual tetrahedra, X stabilizers are primal faces/dual edges, and
    Z stabilizers are primal cells/dual vertices.  The visual representation
    maps qubits to primal vertices in a regular tetrahedron and draws the
    stabilizers as primal faces and cells.
    """

    dimension = 3
    deformation_names: List[str] = []

    _BOUNDARY_BY_SIDE = {
        'left': ('boundary', 'yellow'),
        'right': ('boundary', 'green'),
        'center': ('boundary', 'red'),
        'bottom': ('boundary', 'blue'),
    }

    _COLORS = ('red', 'green', 'blue', 'yellow')

    _TETRAHEDRON_CORNERS = {
        'red': np.array([1.0, 1.0, 1.0], dtype=float),
        'green': np.array([1.0, -1.0, -1.0], dtype=float),
        'blue': np.array([-1.0, 1.0, -1.0], dtype=float),
        'yellow': np.array([-1.0, -1.0, 1.0], dtype=float),
    }

    _COLOR_OFFSETS = {
        'red': np.array([0.22, 0.22, 0.22], dtype=float),
        'green': np.array([0.22, -0.22, -0.22], dtype=float),
        'blue': np.array([-0.22, 0.22, -0.22], dtype=float),
        'yellow': np.array([-0.22, -0.22, 0.22], dtype=float),
    }

    def __init__(
        self,
        L_x: int,
        L_y: int | None = None,
        L_z: int | None = None,
    ):
        if L_x < 3 or L_x % 2 == 0:
            raise ValueError(
                "TetrahedralColor3DCode expects an odd distance d >= 3 as L_x"
            )
        if L_y is not None and L_y != L_x:
            raise ValueError(
                "TetrahedralColor3DCode is tetrahedral: L_y must equal L_x"
            )
        if L_z is not None and L_z != L_x:
            raise ValueError(
                "TetrahedralColor3DCode is tetrahedral: L_z must equal L_x"
            )

        self._lattice_built = False
        self._dual_vertices: List[Node] = []
        self._dual_edges: List[Edge] = []
        self._dual_faces: List[Tuple[Node, Node, Node]] = []
        self._dual_tetrahedra: List[Tetra] = []
        self._vertex_to_tetras: Dict[Node, List[Tetra]] = {}
        self._edge_to_tetras: Dict[Edge, List[Tetra]] = {}
        self._vertex_to_edges: Dict[Node, List[Edge]] = {}
        self._dual_vertex_positions: Dict[Node, np.ndarray] = {}
        self._tetra_positions: Dict[Tetra, np.ndarray] = {}
        self._cell_mesh_cache: Dict[
            Node, Dict[str, List[float] | List[int]]
        ] = {}

        super().__init__(L_x, L_y, L_z)

    @property
    def label(self) -> str:
        return f'Tetrahedral Color d={self.L_x}'

    @property
    def distance(self) -> int:
        return self.L_x

    @property
    def boundary_vertices(self) -> Dict[str, Node]:
        return {
            boundary[1]: boundary
            for boundary in self._BOUNDARY_BY_SIDE.values()
        }

    @property
    def dual_vertices(self) -> List[Node]:
        self._ensure_lattice()
        return self._dual_vertices

    @property
    def dual_edges(self) -> List[Edge]:
        self._ensure_lattice()
        return self._dual_edges

    @property
    def dual_faces(self) -> List[Tuple[Node, Node, Node]]:
        self._ensure_lattice()
        return self._dual_faces

    @property
    def dual_tetrahedra(self) -> List[Tetra]:
        self._ensure_lattice()
        return self._dual_tetrahedra

    @property
    def vertex_to_tetras(self) -> Dict[Node, List[Tetra]]:
        self._ensure_lattice()
        return self._vertex_to_tetras

    @property
    def edge_to_tetras(self) -> Dict[Edge, List[Tetra]]:
        self._ensure_lattice()
        return self._edge_to_tetras

    def get_qubit_coordinates(self) -> Coordinates:
        self._ensure_lattice()
        return list(self._dual_tetrahedra)

    def get_stabilizer_coordinates(self) -> Coordinates:
        self._ensure_lattice()
        interior_vertices = [
            vertex for vertex in self._dual_vertices
            if not self._is_boundary_vertex(vertex)
        ]
        primal_faces = [
            edge for edge in self._dual_edges
            if not self._is_border_edge(edge)
        ]

        return [
            *[
                ('X', edge)
                for edge in primal_faces
            ],
            *[
                ('Z', vertex)
                for vertex in interior_vertices
            ],
        ]

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        pauli, simplex = location
        if pauli == 'X':
            return 'face-x'
        elif pauli == 'Z':
            return f'cell-{self._vertex_color(simplex)}-z'
        else:
            raise ValueError(f"Invalid stabilizer Pauli {pauli}")

    def get_stabilizer(self, location: Tuple) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        pauli, simplex = location
        if pauli == 'X':
            tetras = self._edge_to_tetras[simplex]
        elif pauli == 'Z':
            tetras = self._vertex_to_tetras[simplex]
        else:
            raise ValueError(f"Invalid stabilizer Pauli {pauli}")

        return {
            tetra: pauli
            for tetra in tetras
        }

    def qubit_axis(self, location: Tetra):
        return 'none'

    def get_logicals_x(self) -> List[Operator]:
        self._ensure_lattice()
        boundary = self.boundary_vertices['red']

        return [
            {
                tetra: 'X'
                for tetra in self._vertex_to_tetras[boundary]
            }
        ]

    def get_logicals_z(self) -> List[Operator]:
        self._ensure_lattice()
        boundary = self.boundary_vertices['red']

        return [
            {
                tetra: 'Z'
                for tetra in self._vertex_to_tetras[boundary]
            }
        ]

    def qubit_representation(
        self, location: Tetra, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().qubit_representation(
            location, rotated_picture, json_file=json_file
        )
        rep['location'] = self._tetra_positions[location].tolist()

        return rep

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().stabilizer_representation(
            location, rotated_picture, json_file=json_file
        )

        pauli, simplex = location
        if pauli == 'X':
            mesh = self._face_mesh(simplex)
            rep['location'] = self._edge_position(simplex).tolist()
        elif pauli == 'Z':
            mesh = self._cell_mesh(simplex)
            rep['location'] = self._dual_vertex_positions[simplex].tolist()
        else:
            raise ValueError(f"Invalid stabilizer Pauli {pauli}")

        rep['params']['vertices'] = mesh['vertices']
        rep['params']['triangles'] = mesh['triangles']
        rep['params']['edges'] = mesh['edges']

        return rep

    def _ensure_lattice(self) -> None:
        if self._lattice_built:
            return

        vertices, edges = self._build_dual_graph()
        faces = self._find_cliques(vertices, edges, 3, include_boundary=True)
        tetrahedra = [
            Tetra(tetra)
            for tetra in self._find_cliques(
                vertices, edges, 4, include_boundary=False
            )
        ]

        vertex_to_tetras = {vertex: [] for vertex in vertices}
        edge_to_tetras = {edge: [] for edge in edges}
        vertex_to_edges = {vertex: [] for vertex in vertices}

        for edge in edges:
            u, v = edge
            vertex_to_edges[u].append(edge)
            vertex_to_edges[v].append(edge)

        for tetra in tetrahedra:
            for vertex in tetra.vertices:
                vertex_to_tetras[vertex].append(tetra)
            for edge in combinations(tetra.vertices, 2):
                edge_to_tetras[self._edge(*edge)].append(tetra)

        self._dual_vertices = sorted(vertices)
        self._dual_edges = sorted(edges)
        self._dual_faces = sorted(faces)
        self._dual_tetrahedra = sorted(tetrahedra)
        self._vertex_to_tetras = {
            vertex: sorted(tetras)
            for vertex, tetras in vertex_to_tetras.items()
        }
        self._edge_to_tetras = {
            edge: sorted(tetras)
            for edge, tetras in edge_to_tetras.items()
        }
        self._vertex_to_edges = {
            vertex: sorted(incident_edges)
            for vertex, incident_edges in vertex_to_edges.items()
        }
        self._dual_vertex_positions = {
            vertex: self._dual_vertex_position(vertex)
            for vertex in self._dual_vertices
        }
        self._tetra_positions = self._primal_tetra_positions()
        self._lattice_built = True

    def _build_dual_graph(self) -> Tuple[List[Node], List[Edge]]:
        m = (self.distance - 1) // 2
        edges = set()
        vertices: List[Node] = list(self._BOUNDARY_BY_SIDE.values())
        rgb_layers: Dict[int, Dict[Tuple[int, int], Node]] = {}

        for boundary_a, boundary_b in combinations(vertices, 2):
            edges.add(self._edge(boundary_a, boundary_b))

        for layer in range(m, 0, -1):
            red_offset = (0, 0)
            green_offset = (0, 2)
            yellow_blue_offset = (1, 1)

            red = [
                ('red', col, row, layer)
                for col, row in self._triangular_node_positions(
                    layer, red_offset
                )
            ]
            green = [
                ('green', col, row, layer)
                for col, row in self._triangular_node_positions(
                    layer, green_offset
                )
            ]
            yellow = [
                ('yellow', col, row, layer)
                for col, row in self._triangular_node_positions(
                    layer, yellow_blue_offset
                )
            ]
            blue = [
                ('blue', col, row, layer)
                for col, row in self._triangular_node_positions(
                    layer, yellow_blue_offset
                )
            ]

            vertices.extend(red + green + yellow + blue)
            rgy_layer = {
                vertex[1:3]: vertex
                for vertex in red + green + yellow
            }
            rgb_layer = {
                vertex[1:3]: vertex
                for vertex in red + green + blue
            }
            rgb_layers[layer] = rgb_layer

            self._add_side_boundary_edges(
                edges, layer, red_offset, rgy_layer, 'left'
            )
            self._add_side_boundary_edges(
                edges, layer, green_offset, rgy_layer, 'left'
            )
            self._add_side_boundary_edges(
                edges, layer, yellow_blue_offset, rgb_layer, 'left'
            )

            self._add_side_boundary_edges(
                edges, layer, red_offset, rgy_layer, 'right'
            )
            self._add_side_boundary_edges(
                edges, layer, yellow_blue_offset, rgy_layer, 'right'
            )
            self._add_side_boundary_edges(
                edges, layer, yellow_blue_offset, rgb_layer, 'right'
            )

            self._add_side_boundary_edges(
                edges, layer, green_offset, rgy_layer, 'center'
            )
            self._add_side_boundary_edges(
                edges, layer, yellow_blue_offset, rgy_layer, 'center'
            )
            self._add_side_boundary_edges(
                edges, layer, yellow_blue_offset, rgb_layer, 'center'
            )

            self._add_edges_within_layer(edges, rgy_layer, rgb_layer)

            if layer == m:
                for vertex in rgy_layer.values():
                    edges.add(
                        self._edge(
                            self._BOUNDARY_BY_SIDE['bottom'], vertex
                        )
                    )
            else:
                self._add_edges_to_next_layer(
                    edges,
                    rgy_layer,
                    rgb_layers[layer + 1],
                )

        return vertices, sorted(edges)

    def _add_side_boundary_edges(
        self,
        edges: set[Edge],
        layer: int,
        offset: Tuple[int, int],
        layer_vertices: Dict[Tuple[int, int], Node],
        side: str,
    ) -> None:
        boundary = self._BOUNDARY_BY_SIDE[side]
        for position in self._triangular_node_positions(layer, offset, side):
            edges.add(self._edge(boundary, layer_vertices[position]))

    def _add_edges_within_layer(
        self,
        edges: set[Edge],
        rgy_layer: Dict[Tuple[int, int], Node],
        rgb_layer: Dict[Tuple[int, int], Node],
    ) -> None:
        for (x, y), vertex in rgy_layer.items():
            if vertex[0] == 'yellow':
                edges.add(self._edge(rgb_layer[(x, y)], vertex))

            for x_offset in [-1, 1]:
                for y_offset in [-1, 0, 1]:
                    neighbor = rgy_layer.get((x + x_offset, y + y_offset))
                    self._add_layer_edge(edges, vertex, neighbor)
                    if vertex[0] == 'yellow':
                        self._add_layer_edge(
                            edges, rgb_layer[(x, y)], neighbor
                        )

            for y_offset in [-1, 1]:
                neighbor = (
                    rgy_layer.get((x, y + y_offset))
                    or rgy_layer.get((x, y + 2*y_offset))
                )
                self._add_layer_edge(edges, vertex, neighbor)
                if vertex[0] == 'yellow':
                    self._add_layer_edge(
                        edges, rgb_layer[(x, y)], neighbor
                    )

    def _add_layer_edge(
        self,
        edges: set[Edge],
        vertex: Node,
        neighbor: Node | None,
    ) -> None:
        if neighbor is not None:
            edges.add(self._edge(vertex, neighbor))

    def _add_edges_to_next_layer(
        self,
        edges: set[Edge],
        rgy_layer: Dict[Tuple[int, int], Node],
        next_rgb_layer: Dict[Tuple[int, int], Node],
    ) -> None:
        for (x, y), vertex in rgy_layer.items():
            next_y = y + 2 if vertex[0] == 'red' else y + 1
            edges.add(self._edge(next_rgb_layer[(x, next_y)], vertex))

            if vertex[0] == 'green':
                continue

            for x_offset in [-1, 1]:
                for y_offset in [-1, 0, 1]:
                    neighbor = next_rgb_layer.get(
                        (x + x_offset, next_y + y_offset)
                    )
                    if neighbor is not None and neighbor[0] != vertex[0]:
                        edges.add(self._edge(vertex, neighbor))

            for y_offset in [-1, 1]:
                neighbor = (
                    next_rgb_layer.get((x, next_y + y_offset))
                    or next_rgb_layer.get((x, next_y + 2*y_offset))
                )
                if neighbor is not None and neighbor[0] != vertex[0]:
                    edges.add(self._edge(vertex, neighbor))

    def _find_cliques(
        self,
        vertices: List[Node],
        edges: List[Edge],
        size: int,
        include_boundary: bool,
    ) -> List[Tuple[Node, ...]]:
        order = {
            vertex: index
            for index, vertex in enumerate(sorted(vertices))
        }
        adjacency = {vertex: set() for vertex in vertices}
        for u, v in edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        cliques = []
        for clique in self._extend_clique(
            tuple(), sorted(vertices), adjacency, order, size
        ):
            if include_boundary or not all(
                self._is_boundary_vertex(vertex) for vertex in clique
            ):
                cliques.append(clique)

        return cliques

    def _extend_clique(
        self,
        prefix: Tuple[Node, ...],
        candidates: List[Node],
        adjacency: Dict[Node, set[Node]],
        order: Dict[Node, int],
        target_size: int,
    ) -> Iterable[Tuple[Node, ...]]:
        if len(prefix) == target_size:
            yield prefix
            return

        for index, vertex in enumerate(candidates):
            next_candidates = [
                candidate
                for candidate in candidates[index + 1:]
                if (
                    order[candidate] > order[vertex]
                    and candidate in adjacency[vertex]
                )
            ]

            if prefix:
                prefix_neighbors = set.intersection(
                    *(adjacency[prefix_vertex] for prefix_vertex in prefix)
                )
                next_candidates = [
                    candidate
                    for candidate in next_candidates
                    if candidate in prefix_neighbors
                ]

            if len(prefix) + 1 + len(next_candidates) < target_size:
                continue

            yield from self._extend_clique(
                prefix + (vertex,),
                next_candidates,
                adjacency,
                order,
                target_size,
            )

    def _triangular_node_positions(
        self,
        num_rows: int,
        offset: Tuple[int, int] = (0, 0),
        side: str | None = None,
    ) -> List[Tuple[int, int]]:
        if side is None:
            return [
                (col + offset[0], row + offset[1])
                for row in range(0, 2*num_rows, 2)
                for col in range(-row//2, row//2 + 1, 2)
            ]

        if side == 'left':
            return [
                (-row//2 + offset[0], row + offset[1])
                for row in range(0, 2*num_rows, 2)
            ]
        elif side == 'right':
            return [
                (row//2 + offset[0], row + offset[1])
                for row in range(0, 2*num_rows, 2)
            ]
        elif side == 'center':
            row = 2*num_rows - 2
            return [
                (col + offset[0], row + offset[1])
                for col in range(-row//2, row//2 + 1, 2)
            ]
        else:
            raise ValueError(f"Unknown triangular side {side}")

    def _primal_tetra_positions(self) -> Dict[Tetra, np.ndarray]:
        """Embed qubits in a regular primal tetrahedron.

        Each barycentric coordinate is the shortest face-adjacency distance
        from the qubit tetrahedron to the corresponding colored dual boundary.
        This pins boundary supports to the correct faces and gives the
        d=3 Reed-Muller layout as the first non-trivial instance of the same
        rule used for every odd distance.
        """
        tetra_adjacency = self._tetra_face_adjacency()
        boundary_distances = {
            color: self._tetra_distances_from_boundary(color, tetra_adjacency)
            for color in self._COLORS
        }
        corners = self._regular_tetrahedron_corners()

        positions = {}
        for tetra in self._dual_tetrahedra:
            distances = np.array(
                [
                    boundary_distances[color][tetra]
                    for color in self._COLORS
                ],
                dtype=float,
            )
            distance_sum = float(np.sum(distances))
            if distance_sum == 0:
                weights = np.full(len(self._COLORS), 1/len(self._COLORS))
            else:
                weights = distances / distance_sum

            positions[tetra] = sum(
                weights[index]*corners[color]
                for index, color in enumerate(self._COLORS)
            )

        return positions

    def _tetra_face_adjacency(self) -> Dict[Tetra, List[Tetra]]:
        face_to_tetras: Dict[Tuple[Node, Node, Node], List[Tetra]] = {}
        adjacency = {tetra: set() for tetra in self._dual_tetrahedra}

        for tetra in self._dual_tetrahedra:
            for face in combinations(tetra.vertices, 3):
                sorted_face = tuple(sorted(face))
                face_to_tetras.setdefault(sorted_face, []).append(tetra)

        for incident_tetras in face_to_tetras.values():
            for tetra_a, tetra_b in combinations(incident_tetras, 2):
                adjacency[tetra_a].add(tetra_b)
                adjacency[tetra_b].add(tetra_a)

        return {
            tetra: sorted(neighbors)
            for tetra, neighbors in adjacency.items()
        }

    def _tetra_distances_from_boundary(
        self,
        color: str,
        tetra_adjacency: Dict[Tetra, List[Tetra]],
    ) -> Dict[Tetra, int]:
        boundary = self.boundary_vertices[color]
        distances = {}
        queue = deque()

        for tetra in self._vertex_to_tetras[boundary]:
            distances[tetra] = 0
            queue.append(tetra)

        while queue:
            tetra = queue.popleft()
            for neighbor in tetra_adjacency[tetra]:
                if neighbor in distances:
                    continue

                distances[neighbor] = distances[tetra] + 1
                queue.append(neighbor)

        if len(distances) != len(self._dual_tetrahedra):
            raise ValueError("Dual tetrahedra are not connected")

        return distances

    def _regular_tetrahedron_corners(self) -> Dict[str, np.ndarray]:
        scale = max((self.distance - 1)/2, 1)
        return {
            color: scale*corner
            for color, corner in self._TETRAHEDRON_CORNERS.items()
        }

    def _vertex_color(self, vertex: Node) -> str:
        if self._is_boundary_vertex(vertex):
            return vertex[1]

        return vertex[0]

    def _dual_vertex_position(self, vertex: Node) -> np.ndarray:
        m = (self.distance - 1) // 2
        height = max(m - 1, 1)
        base_centroid_y = 2*height*sqrt(3)/3

        if self._is_boundary_vertex(vertex):
            radius = max(m + 1, 2)
            boundary_positions = {
                self._BOUNDARY_BY_SIDE['left']: np.array(
                    [-radius, base_centroid_y/2, height/2], dtype=float
                ),
                self._BOUNDARY_BY_SIDE['right']: np.array(
                    [radius, base_centroid_y/2, height/2], dtype=float
                ),
                self._BOUNDARY_BY_SIDE['center']: np.array(
                    [0, base_centroid_y + radius, height/2], dtype=float
                ),
                self._BOUNDARY_BY_SIDE['bottom']: np.array(
                    [0, base_centroid_y/2, -radius], dtype=float
                ),
            }
            return boundary_positions[vertex]

        color, col, row, layer = vertex
        layer_height = max(layer - 1, 0)
        layer_centroid_y = 2*layer_height*sqrt(3)/3
        position = np.array(
            [
                float(col),
                row*sqrt(3)/2 + base_centroid_y - layer_centroid_y,
                (m - layer)*sqrt(6)/2,
            ],
            dtype=float,
        )

        return position + self._COLOR_OFFSETS[color]

    def _cell_mesh(self, vertex: Node) -> Dict[str, List[float] | List[int]]:
        if vertex in self._cell_mesh_cache:
            return self._cell_mesh_cache[vertex]

        vertices: List[float] = []
        triangles: List[int] = []
        mesh_edges = set()
        vertex_indices: Dict[Tetra, int] = {}

        def add_vertex(tetra: Tetra) -> int:
            if tetra not in vertex_indices:
                vertex_indices[tetra] = len(vertex_indices)
                vertices.extend(self._tetra_positions[tetra].tolist())

            return vertex_indices[tetra]

        cell_center = np.mean(
            [
                self._tetra_positions[tetra]
                for tetra in self._vertex_to_tetras[vertex]
            ],
            axis=0,
        )

        for edge in self._vertex_to_edges[vertex]:
            face_tetras = [
                tetra
                for tetra in self._edge_to_tetras[edge]
                if vertex in tetra.vertices
            ]
            if len(face_tetras) < 3:
                continue

            ordered_tetras = self._order_tetras_around_cell_face(
                face_tetras, cell_center
            )
            face_indices = [add_vertex(tetra) for tetra in ordered_tetras]
            self._append_oriented_face_triangles(
                triangles, vertices, face_indices, cell_center, mesh_edges
            )

        edges = [
            vertex_index
            for edge in sorted(mesh_edges)
            for vertex_index in edge
        ]
        mesh = {'vertices': vertices, 'triangles': triangles, 'edges': edges}
        self._cell_mesh_cache[vertex] = mesh

        return mesh

    def _append_oriented_face_triangles(
        self,
        triangles: List[int],
        vertices: List[float],
        face_indices: List[int],
        cell_center: np.ndarray,
        mesh_edges: set[Tuple[int, int]] | None = None,
    ) -> None:
        if len(face_indices) < 3:
            return

        points = [
            np.array(vertices[3*index:3*index + 3])
            for index in face_indices
        ]
        face_center = np.mean(points, axis=0)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        outward = face_center - cell_center

        if float(np.dot(normal, outward)) < 0:
            face_indices = [face_indices[0], *reversed(face_indices[1:])]

        if mesh_edges is not None:
            for i, vertex_index in enumerate(face_indices):
                next_vertex_index = face_indices[(i + 1) % len(face_indices)]
                mesh_edges.add(tuple(sorted((vertex_index, next_vertex_index))))

        for i in range(1, len(face_indices) - 1):
            triangles.extend([
                face_indices[0],
                face_indices[i],
                face_indices[i + 1],
            ])

    def _order_tetras_around_cell_face(
        self,
        tetras: List[Tetra],
        cell_center: np.ndarray,
    ) -> List[Tetra]:
        points = np.array([
            self._tetra_positions[tetra]
            for tetra in tetras
        ])
        face_center = np.mean(points, axis=0)
        normal = face_center - cell_center
        norm = np.linalg.norm(normal)
        if norm == 0:
            return sorted(tetras)

        normal = normal / norm
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(normal, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])

        axis_u = np.cross(normal, reference)
        axis_u = axis_u / np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)

        angles = []
        for tetra, point in zip(tetras, points):
            relative = point - face_center
            angle = atan2(
                float(np.dot(relative, axis_v)),
                float(np.dot(relative, axis_u)),
            )
            angles.append((angle, tetra))

        return [
            tetra
            for _, tetra in sorted(angles, key=lambda item: item[0])
        ]

    def _face_mesh(self, edge: Edge) -> Dict[str, List[float] | List[int]]:
        face_tetras = self._edge_to_tetras[edge]
        ordered_tetras = self._order_tetras_around_edge(edge, face_tetras)
        vertices = []

        for tetra in ordered_tetras:
            vertices.extend(self._tetra_positions[tetra].tolist())

        triangles = []
        for index in range(1, len(ordered_tetras) - 1):
            triangles.extend([0, index, index + 1])
        edges = []
        for index in range(len(ordered_tetras)):
            edges.extend([index, (index + 1) % len(ordered_tetras)])

        return {'vertices': vertices, 'triangles': triangles, 'edges': edges}

    def _order_tetras_around_edge(
        self,
        edge: Edge,
        tetras: List[Tetra],
    ) -> List[Tetra]:
        points = np.array([
            self._tetra_positions[tetra]
            for tetra in tetras
        ])
        center = np.mean(points, axis=0)
        normal = (
            self._dual_vertex_positions[edge[1]]
            - self._dual_vertex_positions[edge[0]]
        )
        norm = np.linalg.norm(normal)
        if norm == 0:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / norm

        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(normal, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])

        axis_u = np.cross(normal, reference)
        axis_u = axis_u / np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)

        angles = []
        for tetra, point in zip(tetras, points):
            relative = point - center
            angle = atan2(
                float(np.dot(relative, axis_v)),
                float(np.dot(relative, axis_u)),
            )
            angles.append((angle, tetra))

        return [
            tetra
            for _, tetra in sorted(angles, key=lambda item: item[0])
        ]

    def _edge(self, vertex_a: Node, vertex_b: Node) -> Edge:
        edge = tuple(sorted((vertex_a, vertex_b)))
        return edge  # type: ignore[return-value]

    def _edge_position(self, edge: Edge) -> np.ndarray:
        return np.mean(
            [
                self._dual_vertex_positions[vertex]
                for vertex in edge
            ],
            axis=0,
        )

    def _is_boundary_vertex(self, vertex: Node) -> bool:
        return len(vertex) == 2 and vertex[0] == 'boundary'

    def _is_border_edge(self, edge: Edge) -> bool:
        return all(self._is_boundary_vertex(vertex) for vertex in edge)
