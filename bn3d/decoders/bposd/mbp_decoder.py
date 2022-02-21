import numpy as np
from qecsim.model import Decoder, StabilizerCode, ErrorModel
from typing import Tuple, Dict
import bn3d.bsparse as bsparse
from bn3d.bpauli import bcommute
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


def pauli_to_symplectic(a):
    n = a.shape[0]
    new_a = np.zeros(2*n, dtype='uint8')

    for i in range(n):
        if a[i] == 3 or a[i] == 2:
            new_a[i] = 1
        if a[i] == 1 or a[i] == 2:
            new_a[i + n] = 1

    return new_a


def tanh_prod(a, eps=1e-8):
    """ Square cross product defined in II.B of arXiv:2104.13659"""

    prod = np.prod(np.tanh(a/2))
    if prod >= 1:
        prod -= eps
    elif prod <= -1:
        prod += eps

    # print("Tanh prod", 2 * np.arctanh(np.prod(np.tanh(a/2))))
    return 2 * np.arctanh(np.prod(np.tanh(a/2)))


def log_exp_bias(pauli, gamma, eps=1e-8):
    """ Function lambda defined in II.B of arXiv:2104.13659"""

    denominator = np.sum(np.exp(-gamma), axis=0)
    denominator -= np.exp(-gamma[pauli])
    denominator += eps
    # print("Denominator", denominator)

    return np.log(eps + (1 + np.exp(-gamma[pauli])) / denominator)


def mbp_decoder(H,
                syndrome: np.ndarray,
                p_channel: Dict[str, np.ndarray],
                max_bp_iter=10,
                alpha=0.65,
                beta=0,
                eps=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Belief propagation decoder.
    It returns a probability for each qubit to have had an error
    """

    assert bsparse.is_sparse(H)

    H_pauli = symplectic_to_pauli(H)

    n_stabs, n_qubits = H_pauli.shape

    print("N, M :", n_qubits, n_stabs)

    # Create channel log ratios lambda
    lambda_channel = np.log((p_channel[0] + eps) / p_channel[1:])
    print("Lambda channel", lambda_channel)

    # Initialize vertical messages gamma
    gamma_q2s = np.zeros((3, n_qubits, n_stabs))
    for n in range(n_qubits):
        neighboring_stabs = H_pauli[:, n].nonzero()[0]
        for m in neighboring_stabs:
            for w in range(3):
                gamma_q2s[w, n, m] = lambda_channel[w, n]

    # Initialize horizontal messages delta

    delta_s2q = np.zeros((n_stabs, n_qubits))

    # Main loop
    for iter in range(max_bp_iter):
        print(f"\nIter {iter+1} / {max_bp_iter}")

        # ---------------- Horizontal step ----------------
        for m in range(n_stabs):
            neighboring_qubits = H_pauli[m].nonzero()[1]

            for n in neighboring_qubits:
                lambda_neighbor = np.array([log_exp_bias(H_pauli[m, n_prime]-1, gamma_q2s[:, n_prime, m])
                                            for n_prime in neighboring_qubits if n_prime != n])

                delta_s2q[m, n] = (-1)**syndrome[m] * tanh_prod(lambda_neighbor)

        # print("Delta\n", delta_s2q)

        # ---------------- Vertical step ----------------

        gamma_q = np.zeros((3, n_qubits))
        for n in range(n_qubits):
            neighboring_stabs = H_pauli[:, n].nonzero()[0]
            # print(f"Neighbor of {n}", neighboring_stabs)

            for w in range(3):
                sum_same_pauli = np.sum([delta_s2q[m, n] for m in neighboring_stabs if H_pauli[m, n] == w + 1])
                sum_diff_pauli = np.sum([delta_s2q[m, n] for m in neighboring_stabs if H_pauli[m, n] != w + 1])

                # print("Diff pauli", sum_diff_pauli)

                gamma_q[w, n] = lambda_channel[w, n] + 1 / alpha * sum_diff_pauli - beta * sum_same_pauli

        # ---------------- Hard decision ----------------

        # print("Gamma\n", gamma_q)

        correction = np.zeros(n_qubits, dtype='uint8')

        for n in range(n_qubits):
            if not np.all(gamma_q[:, n] > 0):
                correction[n] = np.argmin(gamma_q[:, n]) + 1

        correction_symplectic = pauli_to_symplectic(correction)

        new_syndrome = bcommute(H, correction_symplectic)
        if np.all(new_syndrome == syndrome):
            print("Syndrome reached\n")
            break

        # ---------------- Inhibition ----------------

        for n in range(n_qubits):
            neighboring_stabs = H_pauli[:, n].nonzero()[0]
            for m in neighboring_stabs:
                for w in range(3):
                    gamma_q2s[w, n, m] = gamma_q[w, n]
                    if 1 + w != H_pauli[m, n]:
                        gamma_q2s[w, n, m] -= delta_s2q[m, n]

        # print("Gamma\n", gamma_q)
        print("Correction", correction)
        print("Correction symplectic", correction_symplectic)
        print("New syndrome", new_syndrome)
        print("Old syndrome", syndrome)

    return correction_symplectic


class MemoryBeliefPropagationDecoder(Decoder):
    label = 'MBP decoder'

    def __init__(self, error_model: ErrorModel,
                 probability: float,
                 max_bp_iter: int = 10,
                 alpha: float = 0.75,
                 beta: float = 0.01):
        super().__init__()
        self._error_model = error_model
        self._probability = probability
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

        if 'Layered Rotated' in code.label:
            L_x, L_y, L_z = code.size

        n_qubits = code.n_k_d[0]
        syndrome = np.array(syndrome, dtype=int)

        pi, px, py, pz = self.get_probabilities(code)

        probabilities = np.vstack([pi, px, py, pz])

        correction = mbp_decoder(
            code.stabilizers, syndrome, probabilities, max_bp_iter=self._max_bp_iter,
            alpha=self._alpha, beta=self._beta
        )
        correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])

        correction = correction.astype(int)

        return correction


def test_symplectic_to_pauli():
    H = csr_matrix(np.array([[1, 1, 0, 1], [1, 0, 1, 1]]))
    print(symplectic_to_pauli(H))


def test_decoder():
    from bn3d.models import Planar2DCode, ToricCode2D
    from bn3d.bpauli import get_effective_error
    from bn3d.noise import PauliErrorModel
    import time
    rng = np.random.default_rng()

    L = 3
    max_bp_iter = 10
    alpha = 0.75
    code = Planar2DCode(L, L)
    # code = ToricCode2D(L, L)
    code.stabilizers

    probability = 0.2
    r_x, r_y, r_z = [0.3, 0.3, 0.4]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    decoder = MemoryBeliefPropagationDecoder(
        error_model, probability, max_bp_iter=max_bp_iter, alpha=alpha
    )

    # Start timer
    start = time.time()

    n_iter = 1
    for i in range(n_iter):
        print(f"\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, probability=probability, rng=rng)
        error = np.zeros(len(error), dtype='uint8')
        error[0] = 1
        error[4] = 1
        print(error)
        print("Calculate syndrome")
        syndrome = bcommute(code.stabilizers, error)
        print(syndrome)
        print("Decode")
        correction = decoder.decode(code, syndrome)
        print("Get total error")
        total_error = (correction + error) % 2
        print("Get effective error")
        effective_error = get_effective_error(
            total_error, code.logical_xs, code.logical_zs
        )
        print("Check codespace")
        codespace = bool(np.all(bcommute(code.stabilizers, total_error) == 0))
        success = bool(np.all(effective_error == 0)) and codespace
        print("Success:", success)

    print("Average time per iteration", (time.time() - start) / n_iter)


if __name__ == '__main__':
    # test_symplectic_to_pauli()
    test_decoder()
