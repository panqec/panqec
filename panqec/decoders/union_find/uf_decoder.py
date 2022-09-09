import imp
import numpy as np
from pymatching import Matching
from panqec.decoders import BaseDecoder
from panqec.codes import Toric2DCode
from panqec.error_models import BaseErrorModel
from panqec.decoders.union_find.clustering import clustering


class UnionFindDecoder(BaseDecoder):
    """Union Find decoder for 2D Toric Code"""

    label = 'Toric 2D Union Find'

    def __init__(self,
                 code: Toric2DCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)

        syndromes_z = self.code.extract_z_syndrome(syndrome)
        syndromes_x = self.code.extract_x_syndrome(syndrome)

        print(syndromes_z)
        print("We are decoding with union find!!!")
        l = clustering(syndromes_z, self.code.Hz)
        q =[]
        for e in l:
            q += list(e[1])
        q = list(set(q))
        for i in q:
            correction[i] = 1       

        # Load it into the X block of the full bsf.
        # correction[:self.code.n] = correction_x
        # correction[self.code.n:] = correction_z

        return correction
