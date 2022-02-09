from typing import Tuple, Dict
import numpy as np
from qecsim.model import Decoder
from ...models import RotatedPlanarCode3D
from ...models import RotatedPlanar3DPauli
Indexer = Dict[Tuple[int, int, int], int]


class RotatedSweepDecoder3D(Decoder):

    label = 'Rotated Code 3D Sweep Decoder'
    _rng: np.random.Generator
    max_rounds: int

    def __init__(self, seed: int = 0, max_rounds: int = 4):
        self._rng = np.random.default_rng(seed)
        self.max_rounds = max_rounds

    def get_full_size(self, code) -> Tuple[int, ...]:
        all_index = np.array(
            list(code.qubit_index.keys())
            + list(code.vertex_index.keys())
            + list(code.face_index.keys())
        )
        return tuple(all_index.max(axis=0))

    def get_initial_state(
        self, code: RotatedPlanarCode3D, syndrome: np.ndarray
    ) -> Indexer:
        """Get initial cellular automaton state from syndrome."""
        n_faces = len(code.face_index)
        face_syndromes = syndrome[:n_faces]
        signs = dict()
        for face, index in code.face_index.items():
            signs[face] = int(face_syndromes[index])
        return signs

    def decode(
        self, code: RotatedPlanarCode3D, syndrome: np.ndarray
    ) -> np.ndarray:
        """Get Z corrections given measured syndrome."""

        # Maximum number of times to apply sweep rule before giving up round.
        largest_size = 2*int(max(code.size)) + 2
        max_sweeps = 4*largest_size

        # The syndromes represented as an array of 0s and 1s.
        signs = self.get_initial_state(code, syndrome)

        # Keep track of the correction needed.
        correction = RotatedPlanar3DPauli(code)

        # Sweep directions to take
        sweep_directions = [
            (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1),
            (-1, 0, 1), (-1, 0, -1),
            (0, -1, 1), (0, -1, -1),
        ]

        # Keep sweeping in all directions until there are no syndromes.
        i_round = 0
        while any(signs.values()) and i_round < self.max_rounds:
            for sweep_direction in sweep_directions:

                # Initialize the number of sweeps.
                i_sweep = 0

                # Keep sweeping until there are no syndromes.
                while any(signs.values()) and i_sweep < max_sweeps:
                    signs = self.sweep_move(
                        signs, correction, sweep_direction, code
                    )
                    i_sweep += 1
            i_round += 1

        return correction.to_bsf()

    def get_sweep_faces(self, vertex, sweep_direction, code):
        """Get the coordinates of neighbouring faces in sweep direction."""
        x, y, z = vertex
        s_x, s_y, s_z = sweep_direction

        # (s_x, s_y) in [(1, 0), (0, 1)]
        if s_x + s_y > 0:
            x_face = (x + 1, y + 1, z + 1*s_z)

        # (s_x, s_y) in [(-1, 0), (0, -1)]
        else:
            x_face = (x - 1, y - 1, z + 1*s_z)

        # (s_x, s_y) in [(1, 0), (0, -1)]
        if s_x - s_y > 0:
            y_face = (x + 1, y - 1, z + 1*s_z)

        # (s_x, s_y) in [(-1, 0), (0, 1)]
        else:
            y_face = (x - 1, y + 1, z + 1*s_z)

        z_face = (x + 2*s_x, y + 2*s_y, z)
        return x_face, y_face, z_face

    def get_sweep_edges(self, vertex, sweep_direction, code):
        """Get coordinates of neighbouring edges in sweep direction."""
        x, y, z = vertex
        s_x, s_y, s_z = sweep_direction

        if s_x - s_y > 0:
            x_edge = (x + 1, y - 1, z)
        else:
            x_edge = (x - 1, y + 1, z)

        if s_x + s_y > 0:
            y_edge = (x + 1, y + 1, z)
        else:
            y_edge = (x - 1, y - 1, z)

        z_edge = (x, y, z + 1*s_z)

        return x_edge, y_edge, z_edge

    def get_default_direction(self):
        """The default direction when all faces are excited."""
        direction = int(self._rng.choice([0, 1, 2], size=1))
        return direction

    def sweep_move(
        self, signs: Indexer, correction: RotatedPlanar3DPauli,
        sweep_direction: Tuple[int, int, int], code: RotatedPlanarCode3D
    ) -> Indexer:
        """Apply the sweep move once along a particular direciton."""

        flip_locations = []

        # Apply sweep rule on every vertex.
        for vertex in code.vertex_index.keys():

            # Get neighbouring faces and edges in the sweep direction.
            x_face, y_face, z_face = self.get_sweep_faces(
                vertex, sweep_direction, code
            )
            x_edge, y_edge, z_edge = self.get_sweep_edges(
                vertex, sweep_direction, code
            )

            # Check faces and edges are in lattice before proceeding.
            faces_valid = tuple(
                face in code.face_index
                for face in [x_face, y_face, z_face]
            )
            edges_valid = tuple(
                edge in code.qubit_index
                for edge in [x_edge, y_edge, z_edge]
            )
            if all(faces_valid) and all(edges_valid):

                if signs[x_face] and signs[y_face] and signs[z_face]:
                    direction = self.get_default_direction()
                    edge_flip = {0: x_edge, 1: y_edge, 2: z_edge}[direction]
                    flip_locations.append(edge_flip)
                elif signs[y_face] and signs[z_face]:
                    flip_locations.append(x_edge)
                elif signs[x_face] and signs[z_face]:
                    flip_locations.append(y_edge)
                elif signs[x_face] and signs[y_face]:
                    flip_locations.append(z_edge)

            """
            # Boundary case with only 1 face and 2 edges.
            elif sum(faces_valid) == 1 and sum(edges_valid) == 2:
                i_face_valid = faces_valid.index(True)
                i_edge_invalid = edges_valid.index(False)
                i_edge_valid = edges_valid.index(True)

                # Check valid face is orthogonal to missing edge.
                if i_face_valid == i_edge_invalid:
                    valid_face = [x_face, y_face, z_face][i_face_valid]
                    valid_edge = [x_edge, y_edge, z_edge][i_edge_valid]

                    # Flip an edge if the face has a syndrome.
                    if signs[valid_face]:
                        flip_locations.append(valid_edge)
            """

        new_signs = signs.copy()
        for location in flip_locations:
            self.flip_edge(location, new_signs, code)
            correction.site('Z', location)

        return new_signs

    def flip_edge(
        self, edge: Tuple, signs: Indexer, code: RotatedPlanarCode3D
    ):
        """Flip signs at index and update correction."""

        x, y, z = edge

        # Determine the axis the edge is parallel to.
        if z % 2 == 0:
            edge_direction = 'z'
        elif x % 4 == 1:
            if y % 4 == 1:
                edge_direction = 'x'
            elif y % 4 == 3:
                edge_direction = 'y'
        elif x % 4 == 3:
            if y % 4 == 1:
                edge_direction = 'y'
            elif y % 4 == 3:
                edge_direction = 'x'

        # Get the faces adjacent to the edge.
        if edge_direction == 'x':
            faces = [
                (x + 1, y + 1, z),
                (x - 1, y - 1, z),
                (x, y, z + 1),
                (x, y, z - 1),
            ]
        elif edge_direction == 'y':
            faces = [
                (x + 1, y - 1, z),
                (x - 1, y + 1, z),
                (x, y, z + 1),
                (x, y, z - 1),
            ]
        elif edge_direction == 'z':
            faces = [
                (x + 1, y + 1, z),
                (x - 1, y - 1, z),
                (x - 1, y + 1, z),
                (x + 1, y - 1, z),
            ]

        # Only keep faces that are actually on the cut lattice.
        faces = [face for face in faces if face in code.face_index]

        # Flip the state of the faces.
        for face in faces:
            signs[face] = 1 - signs[face]
