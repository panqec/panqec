import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import (
    BaseDecoder, SweepDecoder3D, Toric3DMatchingDecoder
)


class SweepMatchDecoder(BaseDecoder):

    label = 'Toric 3D Sweep + Matching Decoder'
    sweeper: SweepDecoder3D
    matcher: Toric3DMatchingDecoder

    def __init__(self, code, error_model, error_rate):
        super().__init__(code, error_model, error_rate)
        self.sweeper = SweepDecoder3D(code, error_model, error_rate)
        self.matcher = Toric3DMatchingDecoder(code, error_model, error_rate)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self.sweeper.decode(syndrome)
        x_correction = self.matcher.decode(syndrome)

        correction = (x_correction + z_correction) % 2

        return correction
