from typing import Tuple, Dict
import numpy as np
from qecsim.model import Decoder
from ...models import ToricCode3D
from ...models import Toric3DPauli
Indexer = Dict[Tuple[int, int, int], int]


class SweepDecoder3D(Decoder):

    label: str = 'Toric 3D Sweep Decoder'
    _rng: np.random.Generator
    max_sweep_factor: int

    def __init__(self, seed: int = 0, max_sweep_factor: int = 32):
        self._rng = np.random.default_rng(seed)
        self.max_sweep_factor = max_sweep_factor

    def get_face_syndromes(
        self, code: ToricCode3D, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        Z vertex stabiziliers syndromes are discarded for this decoder.
        """
        n_faces = code.Hz.shape[0]
        face_syndromes = full_syndrome[:n_faces]
        return face_syndromes

    def flip_edge(
        self, index: Tuple, signs: Indexer, code: ToricCode3D
    ):
        """Flip signs at index and update correction."""
        x, y, z = index
        edge = tuple(np.mod(index, 2))
        L_x, L_y, L_z = code.size
        limits = (2*L_x, 2*L_y, 2*L_z)

        if edge == (1, 0, 0):
            face_1 = (x, y + 1, z)
            face_2 = (x, y - 1, z)
            face_3 = (x, y, z + 1)
            face_4 = (x, y, z - 1)
        elif edge == (0, 1, 0):
            face_1 = (x, y, z + 1)
            face_2 = (x, y, z - 1)
            face_3 = (x + 1, y, z)
            face_4 = (x - 1, y, z)
        elif edge == (0, 0, 1):
            face_1 = (x + 1, y, z)
            face_2 = (x - 1, y, z)
            face_3 = (x, y + 1, z)
            face_4 = (x, y - 1, z)

        # Impose periodic boundary conditions.
        index_1 = tuple(np.mod(face_1, limits))
        index_2 = tuple(np.mod(face_2, limits))
        index_3 = tuple(np.mod(face_3, limits))
        index_4 = tuple(np.mod(face_4, limits))

        # Flip the signs (well actually 0s and 1s).
        signs[index_1] = 1 - signs[index_1]  # type: ignore
        signs[index_2] = 1 - signs[index_2]  # type: ignore
        signs[index_3] = 1 - signs[index_3]  # type: ignore
        signs[index_4] = 1 - signs[index_4]  # type: ignore

    def get_default_direction(self):
        """The default direction when all faces are excited."""
        direction = int(self._rng.choice([0, 1, 2], size=1))
        return direction

    # TODO: make this more space-efficient, don't store zeros.
    def get_initial_state(
        self, code: ToricCode3D, syndrome: np.ndarray
    ) -> Indexer:
        """Get initial cellular automaton state from syndrome."""
        n_faces = len(code.face_index)
        face_syndromes = syndrome[:n_faces]
        signs = dict()
        for face, index in code.face_index.items():
            signs[face] = int(face_syndromes[index])
        return signs

    def decode(
        self, code: ToricCode3D, syndrome: np.ndarray
    ) -> np.ndarray:
        """Get Z corrections given measured syndrome."""

        # Maximum number of times to sweep before giving up.
        max_sweeps = self.max_sweep_factor*int(max(code.size))

        # The syndromes represented as an array of 0s and 1s.
        signs = self.get_initial_state(code, syndrome)

        # Keep track of the correction needed.
        correction = Toric3DPauli(code)

        # Initialize the number of sweeps.
        i_sweep = 0

        # Keep sweeping until there are no syndromes.
        while any(signs.values()) and i_sweep < max_sweeps:
            signs = self.sweep_move(signs, correction, code)
            i_sweep += 1

        return correction.to_bsf()

    def sweep_move(
        self, signs: Indexer, correction: Toric3DPauli,
        code: ToricCode3D
    ) -> Indexer:
        """Apply the sweep move once."""

        new_signs = signs.copy()

        flip_locations = []

        L_x, L_y, L_z = code.size
        limits = (2*L_x, 2*L_y, 2*L_z)

        # Sweep through every edge.
        for x, y, z in code.vertex_index.keys():

            # Get the syndromes on each face in sweep direction.
            x_face = signs[tuple(np.mod((x, y + 1, z + 1), limits))]  # type: ignore  # noqa: E501
            y_face = signs[tuple(np.mod((x + 1, y, z + 1), limits))]  # type: ignore  # noqa: E501
            z_face = signs[tuple(np.mod((x + 1, y + 1, z), limits))]  # type: ignore  # noqa: E501

            x_edge = tuple(np.mod((x + 1, y, z), limits))
            y_edge = tuple(np.mod((x, y + 1, z), limits))
            z_edge = tuple(np.mod((x, y, z + 1), limits))

            if x_face and y_face and z_face:
                direction = self.get_default_direction()
                flip_locations.append(
                    {0: x_edge, 1: y_edge, 2: z_edge}[direction]
                )
            elif y_face and z_face:
                flip_locations.append(x_edge)
            elif x_face and z_face:
                flip_locations.append(y_edge)
            elif x_face and y_face:
                flip_locations.append(z_edge)

        for location in flip_locations:
            self.flip_edge(location, new_signs, code)
            correction.site('Z', location)

        return new_signs
