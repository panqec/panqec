import numpy as np
from qecsim.model import Decoder, StabilizerCode
from ._sweep_decoder_3d import SweepDecoder3D
from ._pymatching_decoder import Toric3DPymatchingDecoder


class SweepMatchDecoder(Decoder):

    label = 'Toric 3D Sweep Pymatching Decoder'
    _sweeper: SweepDecoder3D
    _matcher: Toric3DPymatchingDecoder

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*code.n_k_d[0], dtype=np.uint)

        # Get the Pymatching Matching object.
        matcher = self.get_matcher(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        vertex_syndromes = self.get_vertex_syndromes(code, syndrome)

        # PyMatching gives only the X correction.
        x_correction = matcher.decode(vertex_syndromes, num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:code.n_k_d[0]] = x_correction

        return correction
