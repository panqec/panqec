from typing import Tuple, Dict
import numpy as np
from panqec.decoders import BaseDecoder
from panqec.codes import Toric3DCode

Indexer = Dict[Tuple[int, int, int], int]
Operator = Dict[Tuple[int, int, int], str]


class SweepDecoder3D(BaseDecoder):

    label: str = 'Toric 3D Sweep Decoder'
    _rng: np.random.Generator
    max_sweep_factor: int

    def __init__(self, error_model, probability,
                 seed: int = 0, max_sweep_factor: int = 32):
        super().__init__(error_model, probability)
        self._rng = np.random.default_rng(seed)
        self.max_sweep_factor = max_sweep_factor

    def get_face_syndromes(
        self, code: Toric3DCode, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.
        Z vertex stabilizers syndromes are discarded for this decoder.
        """
        face_syndromes = code.extract_x_syndrome(full_syndrome)
        return face_syndromes

    def flip_edge(
        self, location: Tuple, signs: np.ndarray, code: Toric3DCode
    ):
        """Flip signs at index and update correction."""
        x, y, z = location
        edge = tuple(np.mod(location, 2))
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
        location_1 = tuple(np.mod(face_1, limits))
        location_2 = tuple(np.mod(face_2, limits))
        location_3 = tuple(np.mod(face_3, limits))
        location_4 = tuple(np.mod(face_4, limits))

        # Flip the signs (well actually 0s and 1s).
        if code.is_stabilizer(location_1):
            signs[code.stabilizer_index[location_1]] = 1 - signs[code.stabilizer_index[location_1]]  # type: ignore
        if code.is_stabilizer(location_2):
            signs[code.stabilizer_index[location_2]] = 1 - signs[code.stabilizer_index[location_2]]  # type: ignore
        if code.is_stabilizer(location_3):
            signs[code.stabilizer_index[location_3]] = 1 - signs[code.stabilizer_index[location_3]]  # type: ignore
        if code.is_stabilizer(location_4):
            signs[code.stabilizer_index[location_4]] = 1 - signs[code.stabilizer_index[location_4]]  # type: ignore

    def get_default_direction(self):
        """The default direction when all faces are excited."""
        direction = int(self._rng.choice([0, 1, 2], size=1))
        return direction

    # TODO: make this more space-efficient, don't store zeros.
    def get_initial_state(
        self, code: Toric3DCode, syndrome: np.ndarray
    ) -> Indexer:
        """Get initial cellular automaton state from syndrome."""
        signs = syndrome.copy()
        signs[code.z_indices] = 0

        return signs

    def decode(
        self, code: Toric3DCode, syndrome: np.ndarray
    ) -> np.ndarray:
        """Get Z corrections given measured syndrome."""

        # Maximum number of times to sweep before giving up.
        max_sweeps = self.max_sweep_factor*int(max(code.size))

        # The syndromes represented as an array of 0s and 1s.
        signs = self.get_initial_state(code, syndrome)

        # Keep track of the correction needed.
        correction = dict()

        # Initialize the number of sweeps.
        i_sweep = 0

        # Keep sweeping until there are no syndromes.
        while any(signs) and i_sweep < max_sweeps:
            signs = self.sweep_move(signs, correction, code)
            i_sweep += 1

        return code.to_bsf(correction)

    def sweep_move(
        self, signs: np.ndarray, correction: Operator,
        code: Toric3DCode
    ) -> np.ndarray:
        """Apply the sweep move once."""

        new_signs = signs.copy()

        flip_locations = []

        L_x, L_y, L_z = code.size
        limits = (2*L_x, 2*L_y, 2*L_z)

        # Sweep through every edge.
        for x, y, z in np.array(code.stabilizer_coordinates)[code.z_indices]:
            # Get the syndromes on each face in sweep direction.
            x_face_loc = tuple(np.mod((x, y + 1, z + 1), limits))
            y_face_loc = tuple(np.mod((x + 1, y, z + 1), limits))
            z_face_loc = tuple(np.mod((x + 1, y + 1, z), limits))
            x_face = code.is_stabilizer(x_face_loc) and signs[code.stabilizer_index[x_face_loc]]  # type: ignore  # noqa: E501
            y_face = code.is_stabilizer(y_face_loc) and signs[code.stabilizer_index[y_face_loc]]  # type: ignore  # noqa: E501
            z_face = code.is_stabilizer(z_face_loc) and signs[code.stabilizer_index[z_face_loc]]  # type: ignore  # noqa: E501

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
            correction[location] = 'Z'

        return new_signs