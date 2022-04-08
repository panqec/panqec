"""
Decoder for Rotated Planar 3D Code using Pymatching.
"""
from typing import Dict
import numpy as np
from panqec.codes import StabilizerCode
from pymatching import Matching
from panqec.codes import Toric3DCode
import panqec.bsparse as bsparse
from ._pymatching_decoder import Toric3DPymatchingDecoder


class RotatedPlanarPymatchingDecoder(Toric3DPymatchingDecoder):
    """Pymatching decoder for decoding point sector of Rotated Planar 3D Code.

    Can decode multiple different codes.
    """

    label = 'Rotated Planar 3D Pymatching'
    _matchers: Dict[str, Matching] = {}
    _n_faces: Dict[str, int] = {}
    _n_qubits: Dict[str, int] = {}

    def __init__(self, error_model, probability):
        super().__init__(error_model, probability)
        self._matchers = {}

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""

        return Matching(code.Hz)

    def get_vertex_syndromes(
        self, code: Toric3DCode, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        X face stabiziliers syndromes are discarded for this decoder.
        """
        vertex_syndromes = code.extract_z_syndrome(full_syndrome)

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
        n_qubits = code.n
        correction[:n_qubits] = x_correction

        return bsparse.from_array(correction)
