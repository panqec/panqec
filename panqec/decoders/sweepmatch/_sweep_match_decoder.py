import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from ._sweep_decoder_3d import SweepDecoder3D
from ._pymatching_decoder import Toric3DPymatchingDecoder
import panqec.bsparse as bsparse


class SweepMatchDecoder(BaseDecoder):

    label = 'Toric 3D Sweep Pymatching Decoder'
    _sweeper: SweepDecoder3D
    _matcher: Toric3DPymatchingDecoder

    def __init__(self, error_model, probability):
        super().__init__(error_model, probability)
        self._sweeper = SweepDecoder3D(error_model, probability)
        self._matcher = Toric3DPymatchingDecoder(error_model, probability)

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self._sweeper.decode(code, syndrome)
        x_correction = self._matcher.decode(code, syndrome)

        correction = x_correction + z_correction
        correction.data %= 2
        correction = correction.astype(np.uint)

        return bsparse.from_array(correction)
