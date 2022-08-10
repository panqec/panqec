"""
Decoder for 3D Toric Code using Matching.
"""
import numpy as np
from panqec.decoders import BaseDecoder
from pymatching import Matching
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel


class Toric3DMatchingDecoder(BaseDecoder):
    """Matching decoder for decoding point sector of 3D Toric Codes,
    based on PyMatching.
    """

    label = 'Toric 3D Matching'

    def __init__(self,
                 code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)

        self.matcher = self.get_matcher()

    def get_matcher(self):
        return Matching(self.code.Hz)

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)

        # Keep only the vertex Z syndrome, discard the rest.
        vertex_syndromes = self.code.extract_z_syndrome(syndrome)

        # PyMatching gives only the X correction.
        x_correction = self.matcher.decode(vertex_syndromes,
                                           num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:self.code.n] = x_correction

        return correction
