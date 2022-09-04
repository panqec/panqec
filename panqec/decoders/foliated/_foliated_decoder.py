"""
Foliated decoder for X noise only.

:Author:
    Eric Huang
"""

from typing import Dict, List
import numpy as np
from pymatching import Matching
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel


class FoliatedMatchingDecoder(BaseDecoder):
    """Corrects X noise on y and z edges by many 2D Toric codes.
    Does not correct any other errors.

    Can handle multiple codes at once.
    """

    label: str = 'Foliated Matching'
    allowed_codes: List[str] = []

    matcher_lists: Dict[str, List[Matching]] = dict()

    def __init__(self, code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        self.matcher: Dict = dict()
        super().__init__(code, error_model, error_rate)

    def get_matcher_list(self) -> List[Matching]:
        """Return a new list of Matching objects for a given code."""

        # The size of the lattice.
        L_x, L_y, L_z = self.code.size

        matcher_list: List[Matching] = []
        vertex_indices = self.get_layer_vertices()
        qubit_indices = self.get_layer_qubits()

        for vertex_index, qubit_index in zip(vertex_indices, qubit_indices):
            H_z_layer = self.code.stabilizer_matrix[
                vertex_index, self.code.n:
            ][:, qubit_index]
            matcher_list.append(Matching(H_z_layer))

        return matcher_list

    def get_layer_x(self) -> List[int]:
        return sorted(set([
            x
            for x, y, z in self.code.stabilizer_coordinates
            if self.code.stabilizer_type((x, y, z)) == 'vertex'
        ]))

    def get_layer_vertices(self) -> List[List[int]]:
        """List of vertex indices for each layer."""
        vertex_indices: List[List[int]] = []
        for x in self.get_layer_x():
            vertex_indices.append([
                index
                for index, location in enumerate(
                    self.code.stabilizer_coordinates
                )
                if self.code.stabilizer_type(location) == 'vertex'
                and location[0] == x
            ])
        return vertex_indices

    def get_layer_qubits(self) -> List[List[int]]:
        """List of qubit indices for each layer."""
        qubit_indices: List[List[int]] = []
        for x in self.get_layer_x():
            qubit_indices.append([
                index
                for index, location in enumerate(self.code.qubit_coordinates)
                if self.code.qubit_axis(location) in ['y', 'z']
                and location[0] == x
            ])
        return qubit_indices

    def decode(
        self, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)
        x_correction = np.zeros(self.code.n, dtype=np.uint)

        # Get the Matching Matching object.
        matcher_list = self.get_matcher_list()
        vertex_indices = self.get_layer_vertices()
        qubit_indices = self.get_layer_qubits()

        # Keep only the vertex Z measurement syndrome, discard the rest.
        # Extract the syndrome for each layer.

        # Do matching for each layer.
        for matcher, vertex_index, qubit_index in zip(
            matcher_list, vertex_indices, qubit_indices
        ):
            layer_syndrome = syndrome[vertex_index]
            layer_correction = matcher.decode(
                layer_syndrome, num_neighbours=None
            )
            x_correction[qubit_index] = layer_correction

        # PyMatching gives only the X block correction.
        # Load it into the X block of the full bsf.
        correction[:self.code.n] = x_correction

        return correction
