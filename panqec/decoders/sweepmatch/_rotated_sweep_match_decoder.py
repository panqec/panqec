from ._rotated_sweep_decoder import RotatedSweepDecoder3D
from ._sweep_match_decoder import SweepMatchDecoder
from ._rotated_planar_match_decoder import RotatedPlanarMatchingDecoder
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel


class RotatedSweepMatchDecoder(SweepMatchDecoder):

    label = 'Rotated Planar Code 3D Sweep Matching Decoder'

    def __init__(self, code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float,
                 max_rounds=32):
        super().__init__(code, error_model, error_rate)
        self.sweeper = RotatedSweepDecoder3D(
            code, error_model, error_rate, max_rounds=max_rounds
        )
        self.matcher = RotatedPlanarMatchingDecoder(
            code, error_model, error_rate
        )
