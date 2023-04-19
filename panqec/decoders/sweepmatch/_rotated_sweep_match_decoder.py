from panqec.decoders import (
    BaseDecoder, RotatedSweepDecoder3D, MatchingDecoder
)
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel
import numpy as np


class RotatedSweepMatchDecoder(BaseDecoder):

    label = 'Rotated Planar Code 3D Sweep Matching Decoder'
    allowed_codes = ["RotatedToric3DCode", "RotatedPlanar3DCode"]

    def __init__(self, code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float,
                 max_rounds=32):
        super().__init__(code, error_model, error_rate)

        self.max_rounds = max_rounds

        self.sweeper = RotatedSweepDecoder3D(
            code, error_model, error_rate, max_rounds=max_rounds
        )
        self.matcher = MatchingDecoder(
            code, error_model, error_rate, 'X'
        )

    @property
    def params(self) -> dict:
        return {
            'max_rounds': self.max_rounds
        }

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self.sweeper.decode(syndrome)
        x_correction = self.matcher.decode(syndrome)

        correction = (x_correction + z_correction) % 2

        return correction
