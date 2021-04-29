import numpy as np
from qecsim.model import Decoder, StabilizerCode
from ._sweep_decoder_3d import SweepDecoder3D
from ._pymatching_decoder import Toric3DPymatchingDecoder


class SweepMatchDecoder(Decoder):

    label = 'Toric 3D Sweep Pymatching Decoder'
    _sweeper: SweepDecoder3D
    _matcher: Toric3DPymatchingDecoder

    def __init__(self):
        self._sweeper = SweepDecoder3D()
        self._matcher = Toric3DPymatchingDecoder()

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self._sweeper.decode(code, syndrome)
        x_correction = self._matcher.decode(code, syndrome)

        correction = (z_correction + x_correction) % 2
        return correction
