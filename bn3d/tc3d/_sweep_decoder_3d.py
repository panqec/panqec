import itertools
from typing import Tuple
import numpy as np
from qecsim.model import Decoder
from ._toric_code_3d import ToricCode3D
from ._toric_3d_pauli import Toric3DPauli


class SweepDecoder3D(Decoder):

    label: str = 'Toric 3D Sweep Decoder'

    def get_face_syndromes(
        self, code: ToricCode3D, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        Z vertex stabiziliers syndromes are discarded for this decoder.
        """
        n_faces = int(np.product(code.shape))
        face_syndromes = full_syndrome[:n_faces]
        return face_syndromes

    def flip_edge(
        self, index: Tuple, signs: np.ndarray
    ):
        """Flip signs at index and update correction."""
        edge, L_x, L_y, L_z = index

        # The two orthogonal edge directions.
        ortho_edge_1 = (edge + 1) % 3
        ortho_edge_2 = (edge + 2) % 3

        # Get the indices of the 4 faces surrounding an edge.
        # The faces in each orthogonal direction.
        face_1 = (ortho_edge_1, L_x, L_y, L_z)
        face_2 = (ortho_edge_2, L_x, L_y, L_z)

        # The other face shifted.
        face_3 = np.array(face_1)
        face_3[1 + ortho_edge_2] -= 1

        # yet another face shifted.
        face_4 = np.array(face_2)
        face_4[1 + ortho_edge_1] -= 1

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

    def get_default_direction(self, code: ToricCode3D):
        """The default direction when all faces are excited."""
        return code.X_AXIS

    def decode(self, code: ToricCode3D, syndrome: np.ndarray) -> np.ndarray:
        """Get Z corrections given measured syndrome."""
        default_direction = self.get_default_direction(code)

        signs = np.reshape(
            self.get_face_syndromes(code, syndrome),
            newshape=code.shape
        )
        correction = Toric3DPauli(code)

        signs_0 = signs.copy()
        i = 0
        # Keep sweeping until there are no syndromes.
        while np.any(signs):
            i += 1

            if i > 1000:
                raise Exception()
            signs_0 = signs.copy()
            signs = self.sweep_move(signs, correction, default_direction)
            if np.all(signs_0 == signs):
                import pdb; pdb.set_trace()

        return correction.to_bsf()

    def sweep_move(
        self, signs: np.ndarray, correction: Toric3DPauli,
        default_direction: int = 0
    ) -> np.ndarray:
        """Apply the sweep move once."""

        new_signs = signs.copy()

        ranges = [range(length) for length in signs.shape[1:]]
        flip_locations = []

        # Sweep through every edge.
        for L_x, L_y, L_z in itertools.product(*ranges):

            # Get the syndromes on each face in sweep direction.
            x_face = signs[0, L_x, L_y, L_z]
            y_face = signs[1, L_x, L_y, L_z]
            z_face = signs[2, L_x, L_y, L_z]

            if x_face and y_face and z_face:
                flip_locations.append((default_direction, L_x, L_y, L_z))
            elif y_face and z_face:
                flip_locations.append((0, L_x, L_y, L_z))
            elif x_face and z_face:
                flip_locations.append((1, L_x, L_y, L_z))
            elif x_face and y_face:
                flip_locations.append((2, L_x, L_y, L_z))

        for location in flip_locations:
            self.flip_edge(location, new_signs)
            correction.site('Z', location)

        return new_signs
