import numpy as np
from pymatching import Matching
from panqec.decoders import BaseDecoder
from panqec.codes import Toric2DCode
from panqec.error_models import BaseErrorModel


class Toric2DMatchingDecoder(BaseDecoder):
    """Matching decoder for 2D Toric Code, based on PyMatching"""

    label = 'Toric 2D Matching'

    def __init__(self, code: Toric2DCode, error_model: BaseErrorModel):
        super().__init__(code, error_model)
        self.matcher_z = Matching(self.code.Hz)
        self.matcher_x = Matching(self.code.Hx)

    def decode(self, syndrome: np.ndarray, error_rate: float) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        syndromes_z = self.code.extract_z_syndrome(syndrome)
        syndromes_x = self.code.extract_x_syndrome(syndrome)

        # Match each block using corresponding syndrome but applying correction
        # on the other block.
        correction_x = self.matcher_z.decode(syndromes_z, num_neighbours=None)
        correction_z = self.matcher_x.decode(syndromes_x, num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:self.code.n] = correction_x
        correction[self.code.n:] = correction_z

        return correction
