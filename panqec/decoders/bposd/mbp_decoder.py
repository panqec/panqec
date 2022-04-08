import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from typing import Tuple, Dict
import panqec.bsparse as bsparse
from panqec.bpauli import bcommute
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

    for i in range(n):
        if not reverse:
            if a[i] == 1 or a[i] == 2:
                new_a[i] = 1
            if a[i] == 3 or a[i] == 2:
                new_a[i + n] = 1
        else:
            if a[i] == 3 or a[i] == 2:
                new_a[i] = 1
            if a[i] == 1 or a[i] == 2:
                new_a[i + n] = 1

    return new_a


def tanh_prod(a, eps=1e-8):
    """ Square cross product defined in II.B of arXiv:2104.13659"""

    prod = np.prod(np.tanh(a/2))
    if prod >= 1:
        prod = 1 - eps
    elif prod <= -1:
        prod = -1 + eps

    # print("Tanh prod", 2 * np.arctanh(np.prod(np.tanh(a/2))))
    return 2 * np.arctanh(prod)


def log_exp_bias(pauli, gamma, eps=1e-12):
    """ Function lambda defined in II.B of arXiv:2104.13659"""

    denominator = np.sum(np.exp(-gamma), axis=0)
    denominator -= np.exp(-gamma[pauli])
    if denominator <= 0:
        denominator = eps
    # print("Denominator", denominator)

    return np.log(eps + (1 + np.exp(-gamma[pauli])) / denominator)


def mbp_decoder(H,
                syndrome: np.ndarray,
                p_channel: Dict[str, np.ndarray],
                max_bp_iter=10,
                alpha=0.75,
                beta=0,
                eps=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Belief propagation with memory effect decoder (from arXiv:2104.13659)
    It returns a correction in the binary symplectic format
    """

    # =================== Preprocess parity-check matrix ===================

    # Convert it to a matrix over GF(4), where each element is in [0,4]
    H_pauli = symplectic_to_pauli(H)
    n_stabs, n_qubits = H_pauli.shape

    # Easy access to the neighboring qubits of each stabilizer
    neighboring_qubits = [H_pauli[m].nonzero()[1] for m in range(H_pauli.shape[0])]

    # Easy access to the neighboring stabilizers of each qubit
    neighboring_stabs = [H_pauli[:, n].nonzero()[0] for n in range(H_pauli.shape[1])]

    # ======================= Initialize BP variables =======================

    # Create channel log ratios
    lambda_channel = np.log((p_channel[0] + eps) / p_channel[1:])

    # Initialize [qubit to stabilizer] messages (gamma)
    gamma_q2s = np.zeros((3, n_qubits, n_stabs))
    for n in range(n_qubits):
        for m in neighboring_stabs[n]:
            for w in range(3):
                gamma_q2s[w, n, m] = lambda_channel[w, n]

    # Initialize [stabilizer to qubit] messages (delta)
    delta_s2q = np.zeros((n_stabs, n_qubits))

    # ============================ BP iterations ============================

    for iter in range(max_bp_iter):
        print(f"\nIter {iter+1} / {max_bp_iter}")

        gamma_q = np.zeros((3, n_qubits))
        for n in range(n_qubits):
            # ------------ Stabilizer to qubit update (prod-sum) ------------

            for m in neighboring_stabs[n]:
                lambda_neighbor = np.array([log_exp_bias(H_pauli[m, n_prime]-1, gamma_q2s[:, n_prime, m])
                                            for n_prime in neighboring_qubits[m] if n_prime != n])

                delta_s2q[m, n] = (-1)**syndrome[m] * tanh_prod(lambda_neighbor)

            # ------------------ Qubit to stabilizer update ------------------

            for w in range(3):
                sum_same_pauli = np.sum([delta_s2q[m, n]
                                         for m in neighboring_stabs[n] if H_pauli[m, n] == w + 1])

                sum_diff_pauli = np.sum([delta_s2q[m, n]
                                         for m in neighboring_stabs[n] if H_pauli[m, n] != w + 1])

                gamma_q[w, n] = lambda_channel[w, n] + 1 / alpha * sum_diff_pauli - beta * sum_same_pauli

                # Inhibition loop
                for m in neighboring_stabs[n]:
                    gamma_q2s[w, n, m] = gamma_q[w, n]
                    if 1 + w != H_pauli[m, n]:
                        gamma_q2s[w, n, m] -= delta_s2q[m, n]

        # -------------------------- Hard decision --------------------------

        correction = np.zeros(n_qubits, dtype='uint8')

        for n in range(n_qubits):
            if not np.all(gamma_q[:, n] > 0):
                correction[n] = np.argmin(gamma_q[:, n]) + 1

        correction_symplectic = pauli_to_symplectic(correction)

        # ------------------ Break loop if syndrome reached ------------------

        new_syndrome = bcommute(correction_symplectic, H)
        if np.all(new_syndrome == syndrome):
            print("Syndrome reached\n")
            break

    correction_symplectic = pauli_to_symplectic(correction, reverse=True)

    return correction_symplectic


class MemoryBeliefPropagationDecoder(BaseDecoder):
    label = 'MBP decoder'

    def __init__(self, error_model: BaseErrorModel,
                 probability: float,
                 max_bp_iter: int = 10,
                 alpha: float = 0.4,
                 beta: float = 0.01):
        super().__init__(error_model, probability)
        self._max_bp_iter = max_bp_iter
        self._alpha = alpha
        self._beta = beta

        self._x_decoder: Dict = dict()
        self._z_decoder: Dict = dict()
        self._decoder: Dict = dict()

    def get_probabilities(
        self, code: StabilizerCode
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        pi, px, py, pz = self._error_model.probability_distribution(
            code, self._probability
        )

        return pi, px, py, pz

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        n_qubits = code.n
        syndrome = np.array(syndrome, dtype=int)

        pi, px, py, pz = self.get_probabilities(code)

        probabilities = np.vstack([pi, px, py, pz])

        correction = mbp_decoder(
            code.stabilizer_matrix, syndrome, probabilities, max_bp_iter=self._max_bp_iter,
            alpha=self._alpha, beta=self._beta
        )
        correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])

        return bsparse.from_array(correction)


def test_symplectic_to_pauli():
    H = csr_matrix(np.array([[1, 1, 0, 1], [1, 0, 1, 1]]))
    print(symplectic_to_pauli(H))


def test_decoder():
    from panqec.codes import Toric2DCode
    from panqec.bpauli import get_effective_error
    from panqec.error_models import PauliErrorModel
    import time

    L = 20
    max_bp_iter = 20
    alpha = 0.4

    code = Toric2DCode(L, L)

    probability = 0.3
    r_x, r_y, r_z = [0.333, 0.333, 0.334]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    decoder = MemoryBeliefPropagationDecoder(
        error_model, probability, max_bp_iter=max_bp_iter, alpha=alpha
    )

    # Start timer
    start = time.time()

    n_iter = 1
    for i in range(n_iter):
        print(f"\n\nRun {code.label} {i}...")
        print("Generate errors")
        # error = error_model.generate(code, probability=probability, rng=rng)
        error = bsparse.zero_row(2*code.n)
        error[0, 10] = 1
        error[0, 52] = 1
        error[0, 58] = 1

        print("Calculate syndrome")
        syndrome = bcommute(code.stabilizer_matrix, error)

        print("Decode")
        correction = decoder.decode(code, syndrome)

        print("Get total error")
        total_error = correction + error
        total_error.data %= 2

        print("Get effective error")
        effective_error = get_effective_error(
            total_error, code.logicals_x, code.logicals_z
        )

        print("Check codespace")
        codespace = bool(np.all(bcommute(code.stabilizer_matrix, total_error) == 0))
        success = bool(np.all(effective_error == 0)) and codespace

        print("Success:", success)

    print("Average time per iteration", (time.time() - start) / n_iter)


if __name__ == '__main__':
    test_decoder()
