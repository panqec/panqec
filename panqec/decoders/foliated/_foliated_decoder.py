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


class FoliatedMatchingDecoder(BaseDecoder):
    """Corrects X noise on y and z edges by many 2D Toric codes.
    Does not correct any other errors.

    Can handle multiple codes at once.
    """

    label: str = 'Foliated PyMatching'
    _matcher_lists: Dict[str, List[Matching]] = dict()

    def __init__(self):
        self._matchers = dict()

    def new_matcher_list(self, code: StabilizerCode) -> List[Matching]:
        """Return a new list of Matching objects for a given code."""

        # The size of the lattice.
        L_x, L_y, L_z = code.size

        matcher_list: List[Matching] = []
        vertex_indices = self.get_layer_vertices(code)
        qubit_indices = self.get_layer_qubits(code)

        for vertex_index, qubit_index in zip(vertex_indices, qubit_indices):
            H_z_layer = code.stabilizer_matrix[
                vertex_index, code.n:
            ][:, qubit_index]
            matcher_list.append(Matching(H_z_layer))

        return matcher_list

    def get_matcher_list(self, code: StabilizerCode) -> List[Matching]:
        """Get the list of matchers given the code.

        Matching objects are only instantiated once for the same code.
        """

        # Only instantiate a new Matching object if the code hasn't been seen
        # before.
        if code.label not in self._matchers:
            self._matchers[code.label] = self.new_matcher_list(code)
        return self._matchers[code.label]

    def get_layer_x(self, code: StabilizerCode) -> List[int]:
        return sorted(set([
            x
            for x, y, z in code.stabilizer_coordinates
            if code.stabilizer_type((x, y, z)) == 'vertex'
        ]))

    def get_layer_vertices(self, code: StabilizerCode) -> List[List[int]]:
        """List of vertex indices for each layer."""
        vertex_indices: List[List[int]] = []
        for x in self.get_layer_x(code):
            vertex_indices.append([
                index
                for index, location in enumerate(code.stabilizer_coordinates)
                if code.stabilizer_type(location) == 'vertex'
                and location[0] == x
            ])
        return vertex_indices

    def get_layer_qubits(self, code: StabilizerCode) -> List[List[int]]:
        """List of qubit indices for each layer."""
        qubit_indices: List[List[int]] = []
        for x in self.get_layer_x(code):
            qubit_indices.append([
                index
                for index, location in enumerate(code.qubit_coordinates)
                if code.qubit_axis(location) in ['y', 'z']
                and location[0] == x
            ])
        return qubit_indices

    def decode(
        self, code: StabilizerCode, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*code.n, dtype=np.uint)
        x_correction = np.zeros(code.n, dtype=np.uint)

        # Get the Pymatching Matching object.
        matcher_list = self.get_matcher_list(code)
        vertex_indices = self.get_layer_vertices(code)
        qubit_indices = self.get_layer_qubits(code)

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
        correction[:code.n] = x_correction

        return correction
