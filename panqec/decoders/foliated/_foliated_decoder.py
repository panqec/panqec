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
from ...codes import Toric3DCode


class FoliatedMatchingDecoder(BaseDecoder):
    """Corrects X noise on y and z edges by many 2D Toric codes.

    Can handle multiple codes at once.
    """

    label: str = 'Foliated PyMatching'

    label = 'Toric 3D Pymatching'
    _matcher_lists: Dict[str, List[Matching]] = {}
    _n_faces: Dict[str, int] = {}

    def __init__(self):
        self._matchers = {}
        self._n_faces = {}

    def new_matcher_list(self, code: StabilizerCode) -> List[Matching]:
        """Return a new list of Matching objects for a given code."""
        # Get the number of X stabilizers (faces).
        n_faces = int(np.product(code.shape))
        self._n_faces[code.label] = n_faces

        # Qubits live on edges.
        n_qubits = code.n

        # The size of the lattice.
        L_x, L_y, L_z = code.size

        # Only keep the Z vertex stabilizers and Z block.
        # First index is the vertex and second index is the qubit (edge).
        H_z = code.stabilizer_matrix[n_faces:, n_qubits:]

        # Reshape into ndarray with indices 0, 1, 2 denoting coordinate of
        # the vertex for each stabilizer generator, and index 3, 4, 5, 6
        # denoting the qubit (edge) for that stabilizer generator.
        H_z_3d = H_z.reshape((L_x, L_y, L_z, 3, L_x, L_y, L_z))

        matcher_list: List[Matching] = []

        for x in range(L_x):
            H_z_2d = np.zeros((L_y, L_z, 2, L_y, L_z), dtype=np.uint)
            H_z_2d[:, :, 0, :, :] = H_z_3d[x, :, :, 1, x, :, :]
            H_z_2d[:, :, 1, :, :] = H_z_3d[x, :, :, 2, x, :, :]
            H_z_layer = H_z_2d.reshape((L_y*L_z, 2*L_y*L_z))
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

    def get_vertex_syndromes(
        self, code: Toric3DCode, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        X face stabiziliers syndromes are discarded for this decoder.
        """
        n_faces = self._n_faces[code.label]
        vertex_syndromes = full_syndrome[n_faces:]
        return vertex_syndromes

    def get_layer_syndromes(
        self, code: Toric3DCode, vertex_syndromes: np.ndarray
    ) -> List[np.ndarray]:
        """List of syndromes for each 2D Toric code layer."""

        # Reshape into 3d array of vertices.
        syndromes_3d = vertex_syndromes.reshape(code.size)

        L_y, L_z = code.size[1:]

        # Take slices at each x.
        layer_syndromes: List[np.ndarray] = []
        for x in range(code.size[0]):
            syndromes_2d = syndromes_3d[x, :, :].copy()
            syndromes_layer = syndromes_2d.reshape(L_y*L_z)
            layer_syndromes.append(syndromes_layer)
        return layer_syndromes

    def decode(self, code: Toric3DCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*code.n, dtype=np.uint)

        # Get the Pymatching Matching object.
        matcher_list = self.get_matcher_list(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        vertex_syndromes = self.get_vertex_syndromes(code, syndrome)

        # Extract the syndrome for each layer.
        layer_syndromes = self.get_layer_syndromes(code, vertex_syndromes)

        # Initialize the correction as a shape (3, L_x, L_y, L_z) array.
        correction_3d = np.zeros(code.shape, dtype=np.uint)

        L_x, L_y, L_z = code.size

        # Do matching for each layer.
        for x in range(L_x):
            layer_correction = matcher_list[x].decode(
                layer_syndromes[x], num_neighbours=None
            )
            correction_2d = layer_correction.reshape((2, L_y, L_z))
            correction_3d[1, x, :, :] = correction_2d[0, :, :]
            correction_3d[2, x, :, :] = correction_2d[1, :, :]

        # Reshape into 1d array.
        x_correction = correction_3d.reshape(code.n)

        # PyMatching gives only the X block correction.
        # Load it into the X block of the full bsf.
        correction[:code.n] = x_correction

        return correction
