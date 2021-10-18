from ._rotated_sweep_decoder import RotatedSweepDecoder3D
from ._pymatching_decoder import Toric3DPymatchingDecoder
from ._sweep_match_decoder import SweepMatchDecoder


class RotatedSweepMatchDecoder(SweepMatchDecoder):

    label = 'Rotated Planar Code 3D Sweep Pymatching Decoder'
    _sweeper: RotatedSweepDecoder3D
    _matcher: Toric3DPymatchingDecoder

    def __init__(self):
        self._sweeper = RotatedSweepDecoder3D()
        self._matcher = Toric3DPymatchingDecoder()
