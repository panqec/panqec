import numpy as np

from panqec.codes import Toric2DCode
from panqec.decoders import BaseDecoder
from panqec.decoders.union_find.uf_support import Support
from panqec.error_models import BaseErrorModel, PauliErrorModel


class UnionFindDecoder(BaseDecoder):
    """Union Find decoder for 2D Toric Code"""

    label = 'Toric 2D Union Find'

    allowed_codes = ["Toric2DCode"]

    @property
    def params(self) -> dict:
        return {
        }

    def __init__(self,
                 code: Toric2DCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)
        # an array with value 1 if that index (of the stabilizer) is a syndrome
        syndromes_z = self.code.extract_z_syndrome(syndrome)
        # an array with value 1 if that index (of the stabilizer) is a syndrome
        syndromes_x = self.code.extract_x_syndrome(syndrome)
        # the parity matrix for z stabilizers (a 2D numpy array)
        Hz = self.code.Hz
        # the parity matrix for z stabilizers (a 2D numpy array)
        Hx = self.code.Hx

        support_x = Support(syndromes_x, Hx)
        support_z = Support(syndromes_z, Hz)

        # Load the correction into the X block of the full bsf ###
        correction[:self.code.n] = support_z.decode()
        correction[self.code.n:] = support_x.decode()

        return correction


if __name__ == "__main__":
    error_model = PauliErrorModel(1/3, 1/3, 1/3)
    code = Toric2DCode(20)
    uf = UnionFindDecoder(code, error_model, 0.3)
    seed = 42
    p = 0.1
    errors = error_model.generate(code, p, rng=np.random.default_rng(seed))
    syndrome = code.measure_syndrome(errors)
    correction = uf.decode(syndrome)

    print(f"correction: {correction}")
