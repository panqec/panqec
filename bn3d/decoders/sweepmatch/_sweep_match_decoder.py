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

        z_correction = self._sweeper.decode(code, syndrome).toarray()[0]
        x_correction = self._matcher.decode(code, syndrome)

        print("x", x_correction)
        print("z", z_correction)

        correction = (x_correction + z_correction) % 2
        correction = correction.astype(np.uint)

        print("xz", correction)

        return correction
