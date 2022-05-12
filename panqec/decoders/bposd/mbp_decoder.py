import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from typing import Tuple, Dict, List
import panqec.bsparse as bsparse
from panqec.bpauli import bcommute
from scipy.sparse import csr_matrix

from mbp import mbp_decoder


PAULI_I = 0
PAULI_X = 1
PAULI_Y = 2
PAULI_Z = 3


def symplectic_to_pauli(H):
    n = H.shape[1] // 2
    m = H.shape[0]

    # new_H = bsparse.zero_matrix((m, n))

    new_H = H[:, :n] + 4*H[:, n:]
    new_H.data[new_H.data == 4] = 3
    new_H.data[new_H.data == 5] = 2

    # rows, cols = H.nonzero()

    # for i in range(len(rows)):
    #     if cols[i] < n:
    #         new_H[rows[i], cols[i]] = PAULI_X
    #     else:
    #         if new_H[rows[i], cols[i] - n] == PAULI_X:
    #             new_H[rows[i], cols[i] - n] = PAULI_Y
    #         else:
    #             new_H[rows[i], cols[i] - n] = PAULI_Z

    return new_H


def pauli_to_symplectic(a, reverse=False):
    n = a.shape[0]
    new_a = np.zeros(2*n, dtype='uint8')

    new_a[:n] = np.logical_or((a == 1), (a == 2)).astype('uint8')
    new_a[n:] = np.logical_or((a == 2), (a == 3)).astype('uint8')

    return new_a


def tanh_prod(a, eps=1e-8):
    """ Square cross product defined in II.B of arXiv:2104.13659"""

    prod = np.prod(np.tanh(a/2))
    if prod >= 1:
        prod = 1 - eps
    elif prod <= -1:
        prod = -1 + eps

    return 2 * np.arctanh(prod)


def log_exp_bias(pauli, gamma, eps=1e-12):
    """ Function lambda defined in II.B of arXiv:2104.13659"""

    denominator = np.sum(np.exp(-gamma), axis=0)
    gamma_pauli = np.choose(pauli, gamma)
    exp_gamma_pauli = np.exp(-gamma_pauli)
    denominator -= exp_gamma_pauli

    return np.log(eps + (1 + exp_gamma_pauli)) - np.log(eps + denominator)


class MemoryBeliefPropagationDecoder(BaseDecoder):
    label = 'MBP decoder'

    def __init__(self,
                 code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float,
                 max_bp_iter: int = None,
                 alpha: float = 0.4,
                 beta: float = 0,
                 bp_method: str = 'min_sum',
                 gamma: float = 1):
        super().__init__(code, error_model, error_rate)

        if max_bp_iter is None:
            self.max_bp_iter = code.n
        else:
            self.max_bp_iter = max_bp_iter

        self.alpha = alpha
        self.beta = beta
        self.bp_method = bp_method
        self.gamma = gamma

        self._x_decoder: Dict = dict()
        self._z_decoder: Dict = dict()
        self._decoder: Dict = dict()

        # Convert it to a matrix over GF(4), where each element is in [0,4]
        self.H_pauli = symplectic_to_pauli(code.stabilizer_matrix)

        pi, px, py, pz = self.get_probabilities()
        self.p_channel = np.vstack([pi, px, py, pz])

        self.decoder = mbp_decoder(self.H_pauli,
                                   error_channel=self.p_channel[1:],
                                   alpha_parameter=self.alpha,
                                   max_iter=self.max_bp_iter,
                                   beta_parameter=0.0,
                                   bp_method=self.bp_method,
                                   gamma_parameter=self.gamma)

    def get_probabilities(self):
        # error_rate = self.error_rate
        error_rate = 0.5

        pi, px, py, pz = self.error_model.probability_distribution(
            self.code,
            error_rate
        )

        return pi, px, py, pz

    # @profile
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        # print("Decode")
        correction = self.decoder.decode(syndrome)
        correction_bsf = pauli_to_symplectic(correction)
        # print("Done")

        return correction_bsf


def test_symplectic_to_pauli():
    H = csr_matrix(np.array([[1, 1, 0, 1], [1, 0, 1, 1]]))
    print(symplectic_to_pauli(H))


def test_decoder():
    from panqec.codes import Toric2DCode, Toric3DCode
    from panqec.bpauli import get_effective_error
    from panqec.error_models import PauliErrorModel
    import time

    L = 15
    max_bp_iter = 10
    alpha = 0.4

    code = Toric3DCode(L)

    error_rate = 0.3
    r_x, r_y, r_z = [0.333, 0.333, 0.334]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    print("Instantiating Decoder")
    decoder = MemoryBeliefPropagationDecoder(
        code, error_model, error_rate, max_bp_iter=max_bp_iter, alpha=alpha
    )

    rng = np.random.default_rng(42)

    # Start timer
    start = time.time()

    n_iter = 1
    for i in range(n_iter):
        print(f"\n\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, error_rate=error_rate, rng=rng)
        # error = np.zeros(2*code.n)
        # error[10] = 1
        # error[52] = 1
        # error[58] = 1

        print("Calculate syndrome")
        syndrome = code.measure_syndrome(error)

        print("Decode")
        correction = decoder.decode(syndrome)

        print("Get total error")
        total_error = (correction + error) % 2

        print("Get effective error")
        effective_error = get_effective_error(
            total_error, code.logicals_x, code.logicals_z
        )

        print("Check codespace")
        codespace = code.in_codespace(total_error)
        success = bool(np.all(effective_error == 0)) and codespace

        print("Success:", success)

    print("Average time per iteration", (time.time() - start) / n_iter)


if __name__ == '__main__':
    test_decoder()
