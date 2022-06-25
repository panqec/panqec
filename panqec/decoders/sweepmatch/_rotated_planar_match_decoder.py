"""
Decoder for Rotated Planar 3D Code using PyMatching.
"""
import numpy as np
from typing import Dict
from pymatching import Matching
from panqec.codes import Toric3DCode
from panqec.decoders import Toric3DMatchingDecoder
from panqec.error_models import BaseErrorModel


class RotatedPlanarMatchingDecoder(Toric3DMatchingDecoder):
    """Matching decoder for decoding point sector of Rotated Planar 3D Code,
    based on PyMatching.
    """

    label = 'Rotated Planar 3D Matching'

    _n_faces: Dict[str, int] = {}
    _n_qubits: Dict[str, int] = {}

    def __init__(self, code: Toric3DCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)

    def get_matcher(self):
        """Return a new Matching object."""

        return Matching(self.code.Hz)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        vertex_syndromes = self.code.extract_z_syndrome(syndrome)

        # PyMatching gives only the X correction.
        x_correction = self.matcher.decode(vertex_syndromes,
                                           num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:self.code.n] = x_correction

        return correction
