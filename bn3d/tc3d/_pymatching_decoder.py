"""
Decoder for 3D Toric Code using Pymatching.
"""
import numpy as np
from qecsim.model import Decoder


class Toric3DPymatchingDecoder(Decoder):

    label = 'Toric 3D Pymatching'

    def decode(self, code, syndrome) -> np.ndarray:
        return np.zeros(2*code.n_k_d[0], dtype=np.uint)
