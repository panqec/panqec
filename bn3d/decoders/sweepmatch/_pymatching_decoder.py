"""
Decoder for 3D Toric Code using Pymatching.
"""
from typing import Dict
import numpy as np
from qecsim.model import Decoder, StabilizerCode
from pymatching import Matching
from ...models import ToricCode3D


class Toric3DPymatchingDecoder(Decoder):
    """Pymatching decoder for decoding point sector of 3D Toric Codes.

    Can decode multiple different codes.
    """

    label = 'Toric 3D Pymatching'
    _matchers: Dict[str, Matching] = {}
    _n_faces: Dict[str, int] = {}

    def __init__(self):
        self._matchers = {}
        self._n_faces = {}

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).
        n_faces = int(3*np.product(code.size))
        self._n_faces[code.label] = n_faces
        n_qubits = code.n_k_d[0]

        # Only keep the Z vertex stabilizers.
        H_z = code.stabilizers[n_faces:, n_qubits:]
        return Matching(H_z)

    def get_matcher(self, code: StabilizerCode) -> Matching:
        """Get the matcher given the code.

        Matching objects are only instantiated once for the same code.
        """

        # Only instantiate a new Matching object if the code hasn't been seen
        # before.
        if code.label not in self._matchers:
            self._matchers[code.label] = self.new_matcher(code)
        return self._matchers[code.label]

    def get_vertex_syndromes(
        self, code: ToricCode3D, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        X face stabiziliers syndromes are discarded for this decoder.
        """
        n_faces = self._n_faces[code.label]
        vertex_syndromes = full_syndrome[n_faces:]
        return vertex_syndromes

    def decode(self, code: ToricCode3D, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*code.n_k_d[0], dtype=np.uint)

        # Get the Pymatching Matching object.
        matcher = self.get_matcher(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        vertex_syndromes = self.get_vertex_syndromes(code, syndrome)

        # PyMatching gives only the X correction.
        x_correction = matcher.decode(vertex_syndromes, num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:code.n_k_d[0]] = x_correction

        return correction
