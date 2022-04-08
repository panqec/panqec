from ._rotated_sweep_decoder import RotatedSweepDecoder3D
from ._sweep_match_decoder import SweepMatchDecoder
from ._rotated_planar_pymatching_decoder import RotatedPlanarPymatchingDecoder


class RotatedSweepMatchDecoder(SweepMatchDecoder):

    label = 'Rotated Planar Code 3D Sweep Pymatching Decoder'
    _sweeper: RotatedSweepDecoder3D
    _matcher: RotatedPlanarPymatchingDecoder

    def __init__(self, error_model, probability):
        super().__init__(error_model, probability)
        self._sweeper = RotatedSweepDecoder3D()
        self._matcher = RotatedPlanarPymatchingDecoder()
