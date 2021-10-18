import itertools
from typing import Tuple
import numpy as np
from qecsim.model import Decoder
from ._toric_code_3d import ToricCode3D
from ._toric_3d_pauli import Toric3DPauli


class SweepDecoder3D(Decoder):

    label: str = 'Toric 3D Sweep Decoder'
    _rng: np.random.Generator
    max_sweep_factor: int

    def __init__(self, seed: int = 0, max_sweep_factor: int = 4):
        self._rng = np.random.default_rng(seed)
        self.max_sweep_factor = max_sweep_factor

    def get_face_syndromes(
        self, code: ToricCode3D, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        Z vertex stabiziliers syndromes are discarded for this decoder.
        """
        n_faces = len(code.Hz)
        face_syndromes = full_syndrome[:n_faces]
        return face_syndromes

    def flip_edge(
        self, index: Tuple, signs: np.ndarray
    ):
        """Flip signs at index and update correction."""
        edge, x, y, z = index

        if edge == 0:
            face_1 = (1, x, y, z)
            face_2 = (2, x, y, z)
            face_3 = (1, x, y, z - 1)
            face_4 = (2, x, y - 1, z)
        elif edge == 1:
            face_1 = (2, x, y, z)
            face_2 = (0, x, y, z)
            face_3 = (2, x - 1, y, z)
            face_4 = (0, x, y, z - 1)
        elif edge == 2:
            face_1 = (0, x, y, z)
            face_2 = (1, x, y, z)
            face_3 = (0, x, y - 1, z)
            face_4 = (1, x - 1, y, z)

        # Impose periodic boundary conditions.
        index_1 = tuple(np.mod(face_1, signs.shape))
        index_2 = tuple(np.mod(face_2, signs.shape))
        index_3 = tuple(np.mod(face_3, signs.shape))
        index_4 = tuple(np.mod(face_4, signs.shape))

        # Flip the signs (well actually 0s and 1s).
        signs[index_1] = 1 - signs[index_1]
        signs[index_2] = 1 - signs[index_2]
        signs[index_3] = 1 - signs[index_3]
        signs[index_4] = 1 - signs[index_4]

    def get_default_direction(self):
        """The default direction when all faces are excited."""
        direction = int(self._rng.choice([0, 1, 2], size=1))
        return direction

    def get_sign_array(self, code: ToricCode3D, syndrome: np.ndarray):
        signs = np.reshape(
            self.get_face_syndromes(code, syndrome),
            newshape=code.shape
        )
        return signs

    def decode(
        self, code: ToricCode3D, syndrome: np.ndarray
    ) -> np.ndarray:
        """Get Z corrections given measured syndrome."""

        # Maximum number of times to sweep before giving up.
        max_sweeps = self.max_sweep_factor*int(max(code.size))

        # The syndromes represented as an array of 0s and 1s.
        signs = self.get_sign_array(code, syndrome)

        # Keep track of the correction needed.
        correction = Toric3DPauli(code)

        # Initialize the number of sweeps.
        i_sweep = 0

        # Keep sweeping until there are no syndromes.
        while np.any(signs) and i_sweep < max_sweeps:
            signs = self.sweep_move(signs, correction)
            i_sweep += 1

        return correction.to_bsf()

    def sweep_move(
        self, signs: np.ndarray, correction: Toric3DPauli
    ) -> np.ndarray:
        """Apply the sweep move once."""

        new_signs = signs.copy()

        ranges = [range(length) for length in signs.shape[1:]]
        flip_locations = []

        # Sweep through every edge.
        for x, y, z in itertools.product(*ranges):

            # Get the syndromes on each face in sweep direction.
            x_face = signs[0, x, y, z]
            y_face = signs[1, x, y, z]
            z_face = signs[2, x, y, z]

            if x_face and y_face and z_face:
                direction = self.get_default_direction()
                flip_locations.append((direction, x, y, z))
            elif y_face and z_face:
                flip_locations.append((0, x, y, z))
            elif x_face and z_face:
                flip_locations.append((1, x, y, z))
            elif x_face and y_face:
                flip_locations.append((2, x, y, z))

        for location in flip_locations:
            self.flip_edge(location, new_signs)
            correction.site('Z', location)

        return new_signs
