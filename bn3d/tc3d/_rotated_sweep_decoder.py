import itertools
import numpy as np
from ._sweep_decoder_3d import SweepDecoder3D
from ._rotated_planar_code_3d import RotatedPlanarCode3D
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli


class RotatedSweepDecoder3D(SweepDecoder3D):

    label = 'Rotated Code 3D Sweep Decoder'

    def __init__(self):
        super().__init__()

    def get_sign_array(self, code: RotatedPlanarCode3D, syndrome: np.ndarray):
        signs = np.zeros(code.full_size, dtype=int)
        return signs

    def decode(
        self, code: RotatedPlanarCode3D, syndrome: np.ndarray
    ) -> np.ndarray:
        """Get Z corrections given measured syndrome."""

        # Maximum number of times to sweep before giving up.
        max_sweeps = self.max_sweep_factor*int(max(code.size))

        # The syndromes represented as an array of 0s and 1s.
        signs = self.get_sign_array(code, syndrome)

        # Keep track of the correction needed.
        correction = RotatedPlanar3DPauli(code)

        # Initialize the number of sweeps.
        i_sweep = 0

        # Keep sweeping until there are no syndromes.
        while np.any(signs) and i_sweep < max_sweeps:
            signs = self.sweep_move(signs, correction)
            i_sweep += 1

        return correction.to_bsf()

    def sweep_move(
        self, signs: np.ndarray, correction: RotatedPlanar3DPauli
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
