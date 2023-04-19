from typing import Sequence
import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from typing import Dict
import panqec.bsparse as bsparse
from scipy.sparse import csr_matrix


PAULI_I = 0
PAULI_X = 1
PAULI_Y = 2
PAULI_Z = 3


def symplectic_to_pauli(H):
    n = H.shape[1] // 2
    m = H.shape[0]
    new_H = bsparse.zero_matrix((m, n))
    rows, cols = H.nonzero()
    for i in range(len(rows)):
        if cols[i] < n:
            new_H[rows[i], cols[i]] = PAULI_X
        else:
            if new_H[rows[i], cols[i] - n] == PAULI_X:
                new_H[rows[i], cols[i] - n] = PAULI_Y
            else:
                new_H[rows[i], cols[i] - n] = PAULI_Z
    return new_H


def pauli_to_symplectic(a, reverse=False):
    n = a.shape[0]
    new_a = np.zeros(2*n, dtype='uint8')

    if not reverse:
        new_a[:n] = np.logical_or((a == 1), (a == 2)).astype('uint8')
        new_a[n:] = np.logical_or((a == 2), (a == 3)).astype('uint8')
    else:
        new_a[n:] = np.logical_or((a == 1), (a == 2)).astype('uint8')
        new_a[:n] = np.logical_or((a == 2), (a == 3)).astype('uint8')

    return new_a


def tanh_prod(a, eps=1e-8):
    """ Square cross product defined in II.B of arXiv:2104.13659"""
    prod = np.prod(np.tanh(a/2))
    if prod >= 1:
        prod = 1 - eps
    elif prod <= -1:
        prod = -1 + eps
    return 2 * np.arctanh(prod)


def log_exp_bias(pauli, gamma, eps=1e-12) -> Sequence:
    """ Function lambda defined in II.B of arXiv:2104.13659"""
    denominator = np.sum(np.exp(-gamma), axis=0)
    gamma_pauli = np.choose(pauli, gamma)
    exp_gamma_pauli = np.exp(-gamma_pauli)
    denominator -= exp_gamma_pauli
    return np.log(eps + (1 + exp_gamma_pauli)) - np.log(eps + denominator)


class MemoryBeliefPropagationDecoder(BaseDecoder):
    label = 'MBP decoder'
    allowed_codes = None  # all codes allowed

    def __init__(self,
                 code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float,
                 max_bp_iter: int = 100,
                 alpha: float = 0.4,
                 beta: float = 0):
        super().__init__(code, error_model, error_rate)

        self.max_bp_iter = max_bp_iter
        self.alpha = alpha
        self.beta = beta

        self._x_decoder: Dict = dict()
        self._z_decoder: Dict = dict()

        self._decoder: Dict = dict()
        # Convert it to a matrix over GF(4), where each element is in [0,4]
        self.H = code.stabilizer_matrix.toarray()
        self.H_pauli = symplectic_to_pauli(code.stabilizer_matrix).toarray()
        pi, px, py, pz = self.get_probabilities()
        self.p_channel = np.vstack([pi, px, py, pz])

        # Easy access to the neighboring qubits of each stabilizer
        self.neighboring_qubits = [self.H_pauli[m].nonzero()[0]
                                   for m in range(self.H_pauli.shape[0])]

        # Easy access to the neighboring stabilizers of each qubit
        self.neighboring_stabs = [self.H_pauli[:, n].nonzero()[0]
                                  for n in range(self.H_pauli.shape[1])]

        # ===================== Initialize BP variables ====================

        # Create channel log ratios
        self.lambda_channel = np.log((1 - self.p_channel[1:])
                                     / self.p_channel[1:])

        # Initialize [qubit to stabilizer] messages (gamma)
        n_stabs, n_qubits = self.H_pauli.shape
        self.gamma_q2s = np.zeros((3, n_qubits, n_stabs))
        for n in range(n_qubits):
            for m in self.neighboring_stabs[n]:
                for w in range(3):
                    if 1 + w != self.H_pauli[m, n]:
                        self.gamma_q2s[w, n, m] = self.lambda_channel[w, n]

        # Initialize [stabilizer to qubit] messages (delta)
        self.delta_s2q = np.zeros((n_stabs, n_qubits))

    @property
    def params(self) -> dict:
        return {
            'max_bp_iter': self.max_bp_iter,
            'alpha': self.alpha,
            'beta': self.beta
        }

    def get_probabilities(self):
        error_rate = 0.5
        pi, px, py, pz = self.error_model.probability_distribution(
            self.code,
            error_rate
        )
        return pi, px, py, pz

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        H_pauli = self.H_pauli

        n_stabs, n_qubits = H_pauli.shape

        # ==================== Initialize BP variables ====================

        gamma_q2s = self.gamma_q2s.copy()
        delta_s2q = self.delta_s2q.copy()

        # ========================= BP iterations =========================

        for iter in range(self.max_bp_iter):
            print(f"\nIter {iter+1} / {self.max_bp_iter}")

            gamma_q = np.zeros((3, n_qubits))
            for n in range(n_qubits):
                # --------- Stabilizer to qubit update (prod-sum) ---------

                for m in self.neighboring_stabs[n]:
                    lambda_neighbor = np.array([
                        log_exp_bias(
                            H_pauli[m, n_prime]-1, gamma_q2s[:, n_prime, m]
                        )
                        for n_prime in self.neighboring_qubits[m]
                        if n_prime != n
                    ])
                    lambda_neighbor = np.array(lambda_neighbor)

                    sign = (-1)**syndrome[m]
                    delta_s2q[m, n] = sign * tanh_prod(lambda_neighbor)

                # ----------------- Qubit to stabilizer update ---------------

                for w in range(3):
                    sum_same_pauli = np.sum(
                        delta_s2q[self.neighboring_stabs[n], n][
                            H_pauli[self.neighboring_stabs[n], n] == w + 1
                        ]
                    )
                    sum_diff_pauli = np.sum(
                        delta_s2q[self.neighboring_stabs[n], n][
                            H_pauli[self.neighboring_stabs[n], n] != w + 1
                        ]
                    )

                    gamma_q[w, n] = self.lambda_channel[w, n] \
                        + 1 / self.alpha * sum_diff_pauli \
                        - self.beta * sum_same_pauli

                    # Update qubit to stab messages
                    gamma_q2s[w, n, :] = gamma_q[w, n]

                    # Inhibition loop
                    gamma_q2s[w, n, (1 + w != H_pauli[:, n])] -= delta_s2q[
                        (1 + w != H_pauli[:, n]), n
                    ]

            # ----------------------- Hard decision -----------------------

            correction = np.zeros(n_qubits, dtype='uint8')

            for n in range(n_qubits):
                if not np.all(gamma_q[:, n] > 0):
                    correction[n] = np.argmin(gamma_q[:, n]) + 1

            correction_symplectic = pauli_to_symplectic(correction)

            # --------------- Break loop if syndrome reached ---------------

            new_syndrome = self.code.measure_syndrome(correction_symplectic)
            if np.all(new_syndrome == syndrome):
                # print(f"Syndrome reached in {iter} iterations\n")
                break

        correction_symplectic = pauli_to_symplectic(correction, reverse=True)

        correction = np.concatenate([correction_symplectic[n_qubits:],
                                     correction_symplectic[:n_qubits]])

        return correction


def test_symplectic_to_pauli():
    H = csr_matrix(np.array([[1, 1, 0, 1], [1, 0, 1, 1]]))
    print(symplectic_to_pauli(H))


def test_decoder():
    from panqec.codes import Toric3DCode
    from panqec.error_models import PauliErrorModel
    import time

    L = 4
    max_bp_iter = 10
    alpha = 0.4

    code = Toric3DCode(L)

    error_rate = 0.3
    r_x, r_y, r_z = [0.333, 0.333, 0.334]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    decoder = MemoryBeliefPropagationDecoder(
        code, error_model, error_rate, max_bp_iter=max_bp_iter, alpha=alpha
    )

    # Start timer
    start = time.time()

    n_iter = 1
    for i in range(n_iter):
        print(f"\n\nRun {code.label} {i}...")
        print("Generate errors")
        error = np.zeros(2*code.n)
        error[10] = 1
        error[52] = 1
        error[58] = 1

        print("Calculate syndrome")
        syndrome = code.measure_syndrome(error)

        print("Decode")
        correction = decoder.decode(syndrome)
        print("Get total error")
        total_error = (correction + error) % 2
        print("Get effective error")
        effective_error = code.logical_errors(total_error)

        print("Check codespace")
        codespace = code.in_codespace(total_error)
        success = bool(np.all(effective_error == 0)) and codespace

        print("Success:", success)

    print("Average time per iteration", (time.time() - start) / n_iter)


if __name__ == '__main__':
    test_decoder()
