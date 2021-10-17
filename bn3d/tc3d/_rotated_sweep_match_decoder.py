from ._sweep_match_decoder import SweepMatchDecoder


class RotatedSweepMatchDecoder(SweepMatchDecoder):

    label = 'Rotated Planar Code 3D Sweep Pymatching Decoder'

    def __init__(self):
        super().__init__()
