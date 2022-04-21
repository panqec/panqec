from panqec.decoders import (
    RotatedSweepDecoder3D, SweepMatchDecoder, RotatedPlanarMatchingDecoder
)


class RotatedSweepMatchDecoder(SweepMatchDecoder):

    label = 'Rotated Planar Code 3D Sweep Matching Decoder'
    sweeper: RotatedSweepDecoder3D
    matcher: RotatedPlanarMatchingDecoder

    def __init__(self, code, error_model):
        super().__init__(code, error_model)
        self.sweeper = RotatedSweepDecoder3D()
        self.matcher = RotatedPlanarMatchingDecoder()
