from typing import Tuple, Dict
import numpy as np
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from panqec.codes import Toric3DCode

Indexer = Dict[Tuple[int, int, int], int]
Operator = Dict[Tuple[int, int, int], str]


class SweepDecoder3D(BaseDecoder):

    label: str = 'Toric 3D Sweep Decoder'
    _rng: np.random.Generator
    max_sweep_factor: int

    def __init__(self,
                 code: Toric3DCode,
                 error_model: BaseErrorModel,
                 error_rate: float,
                 seed: int = 0,
                 max_sweep_factor: int = 32):
        super().__init__(code, error_model, error_rate)
        self._rng = np.random.default_rng(seed)
        self.max_sweep_factor = max_sweep_factor

    def flip_edge(
        self, location: Tuple, signs: np.ndarray
    ):
        """Flip signs at index and update correction."""
        x, y, z = location
        edge = tuple(np.mod(location, 2))
        L_x, L_y, L_z = self.code.size
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
        location_1 = tuple(np.mod(face_1, limits))
        location_2 = tuple(np.mod(face_2, limits))
        location_3 = tuple(np.mod(face_3, limits))
        location_4 = tuple(np.mod(face_4, limits))

        # Flip the signs (well actually 0s and 1s).
        for location in [location_1, location_2, location_3, location_4]:
            if self.code.is_stabilizer(location):
                signs[self.code.stabilizer_index[location]] *= -1
                signs[self.code.stabilizer_index[location]] += 1

    def get_default_direction(self):
        """The default direction when all faces are excited."""
        direction = int(self._rng.choice([0, 1, 2], size=1))
        return direction

    def get_initial_state(
        self, syndrome: np.ndarray
    ) -> Indexer:
        """Get initial cellular automaton state from syndrome."""
        signs = syndrome.copy()
        signs[self.code.z_indices] = 0

        return signs

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Get Z corrections given measured syndrome."""

        # Maximum number of times to sweep before giving up.
        max_sweeps = self.max_sweep_factor*int(max(self.code.size))

        # The syndromes represented as an array of 0s and 1s.
        signs = self.get_initial_state(syndrome)

        # Keep track of the correction needed.
        correction = dict()

        # Initialize the number of sweeps.
        i_sweep = 0

        # Keep sweeping until there are no syndromes.
        while any(signs) and i_sweep < max_sweeps:
            signs = self.sweep_move(signs, correction)
            i_sweep += 1

        return self.code.to_bsf(correction)

    def sweep_move(
        self, signs: np.ndarray, correction: Operator,
    ) -> np.ndarray:
        """Apply the sweep move once."""

        new_signs = signs.copy()

        flip_locations = []

        L_x, L_y, L_z = self.code.size
        limits = (2*L_x, 2*L_y, 2*L_z)

        # Sweep through every edge.
        for x, y, z in np.array(self.code.stabilizer_coordinates)[self.code.z_indices]:
            # Get the syndromes on each face in sweep direction.
            x_face_loc = tuple(np.mod((x, y + 1, z + 1), limits))
            y_face_loc = tuple(np.mod((x + 1, y, z + 1), limits))
            z_face_loc = tuple(np.mod((x + 1, y + 1, z), limits))
            x_face = self.code.is_stabilizer(x_face_loc) and signs[self.code.stabilizer_index[x_face_loc]]  # type: ignore  # noqa: E501
            y_face = self.code.is_stabilizer(y_face_loc) and signs[self.code.stabilizer_index[y_face_loc]]  # type: ignore  # noqa: E501
            z_face = self.code.is_stabilizer(z_face_loc) and signs[self.code.stabilizer_index[z_face_loc]]  # type: ignore  # noqa: E501

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
            self.flip_edge(location, new_signs)
            correction[location] = 'Z'

        return new_signs
