"""
Decoder for 3D Toric Code using Pymatching.
"""
from typing import Dict
import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from pymatching import Matching
from panqec.codes import Toric3DCode
import panqec.bsparse as bsparse


class Toric3DPymatchingDecoder(BaseDecoder):
    """Pymatching decoder for decoding point sector of 3D Toric Codes.

    Can decode multiple different codes.
    """

    label = 'Toric 3D Pymatching'
    _matchers: Dict[str, Matching] = {}

    def __init__(self, error_model, probability):
        super().__init__(error_model, probability)
        self._matchers = {}

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).

        # Only keep the Z vertex stabilizers.
        return Matching(code.Hz)

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
        self, code: Toric3DCode, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        X face stabiziliers syndromes are discarded for this decoder.
        """
        vertex_syndromes = full_syndrome[code.z_indices]
        return vertex_syndromes

    def decode(self, code: Toric3DCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*code.n, dtype=np.uint)

        # Get the Pymatching Matching object.
        matcher = self.get_matcher(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        vertex_syndromes = self.get_vertex_syndromes(code, syndrome)

        # PyMatching gives only the X correction.
        x_correction = matcher.decode(vertex_syndromes, num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:code.n] = x_correction

        return bsparse.from_array(correction)
