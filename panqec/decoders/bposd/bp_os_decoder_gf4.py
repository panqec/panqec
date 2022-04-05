import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from typing import Tuple, List, Dict
import galois


GF4 = galois.GF(4)
GF4.display('poly')


def get_rref_gf4(
    A: GF4, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a matrix A and a vector b.
    Return the row echelon form of A and a new vector b,
    modified with the same row operations"""
    n_rows, n_cols = A.shape
    A = A.copy()
    b = b.copy()

    i_pivot = 0
    i_col = 0
    while i_pivot < n_rows and i_col < n_cols:
        i_nonzero_row = i_pivot
        while i_nonzero_row < A.shape[0] - 1 and A[i_nonzero_row, i_col] == 0:
            i_nonzero_row += 1

        if A[i_nonzero_row, i_col] != 0:
            A[[i_pivot, i_nonzero_row]] = A[[i_nonzero_row, i_pivot]]
            b[[i_pivot, i_nonzero_row]] = b[[i_nonzero_row, i_pivot]]

            cond = A[:, i_col] != 0
            cond[i_pivot] = False

            A[cond] = A[cond] + A[i_pivot] / A[i_pivot, i_col] * A[cond, i_col][:, np.newaxis]
            b[cond] = np.logical_xor(b[cond], b[i_pivot])

            i_pivot += 1
        i_col += 1

    return A, b


def solve_rref_gf4(A: GF4, b: np.ndarray) -> np.ndarray:
    """Solve the system Ax=b mod 2, with A in reduced row echelon form"""
    n_rows, n_cols = A.shape
    x = np.zeros(n_rows)

    for i in range(n_rows-1, -1, -1):
        x[i] = (b[i] - A[i].dot(x)) / A[i, i]

    return x % 2


def select_independent_columns(A: GF4) -> List[int]:
    """Select independent columns of a matrix A in reduced row echelon form"""
    n_rows, n_cols = A.shape

    i_col, i_row = 0, 0
    list_col_idx = []
    while i_col < n_cols and i_row < n_rows:
        if A[i_row, i_col] != 0:
            list_col_idx.append(i_col)
            i_row += 1
        i_col += 1

    return list_col_idx


# @profile
def osd_decoder(H: GF4,
                syndrome: np.ndarray,
                bp_proba: np.ndarray) -> np.ndarray:
    """"Ordered Statistics Decoder
    It returns a correction array (1 for a correction and 0 otherwise)
    by inverting the linear system H*e=s
    """

    n_parities, n_data = H.shape

    # Sort columns of H with the probabilities given by the BP algorithm
    sorted_data_indices = list(np.argsort(-bp_proba))
    H_sorted = H[:, sorted_data_indices]

    # Get the reduced row echelon form (rref) of H, to simplify calculations
    H_sorted_rref, syndrome_rref = get_rref_gf4(H_sorted, syndrome)

    # Create a full-rank squared matrix, by selecting independent columns and
    # rows
    selected_col_indices = select_independent_columns(H_sorted_rref)
    selected_row_indices = list(range(len(selected_col_indices)))
    reduced_H_rref = H_sorted_rref[selected_row_indices][
        :, selected_col_indices
    ]
    reduced_syndrome_rref = syndrome_rref[selected_row_indices]

    # Solve the system H*e = s, in its rref
    reduced_correction = solve_rref_gf4(reduced_H_rref, reduced_syndrome_rref)

    # Fill with the 0 the non-selected (-> dependent columns) indices
    sorted_correction = np.zeros(n_data)
    sorted_correction[selected_col_indices] = reduced_correction

    # Rearrange the indices of the correction to take the initial sorting into
    # account.
    correction = np.zeros(n_data)
    correction[sorted_data_indices] = sorted_correction

    return correction


def gf4_to_pauli(gf4_number: int):
    mapping = {0: 'I', 1: 'X', 2: 'Z', 3: 'Y'}

    for n, p in mapping.items():
        if gf4_number == n:
            return p


def commute(w1: GF4, w2: str):
    w1 = gf4_to_pauli(w1)

    return int(w1 == 'I' or w2 == 'I' or w1 == w2)


# @profile
def bp_decoder(H: np.ndarray,
               syndrome: np.ndarray,
               init_proba: Dict[str, np.ndarray],
               max_iter=10,
               eps=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Belief propagation decoder.
    It returns a probability for each qubit to have had an error
    """

    n_parities, n_data = H.shape

    paulis = ['I', 'X', 'Y', 'Z']

    # log_ratio_p = {pauli: np.log((1-p-eps) / (p+eps))
    #                for pauli, p in probabilities.items()}

    # Create tuple with parity indices and data indices
    # Each element (edges_p2d[0, i], edges_p2d[1, i]) is an edge
    # from parity to data
    edges_p2d = np.nonzero(H)
    edges_d2p = (edges_p2d[1], edges_p2d[0])

    # Create messages from parity to data and from data to parity
    # The initialization with np.inf (resp. zero) allows to ignore
    # non-neighboring elements when doing a min (resp. sum)
    message_p2d = np.ones((n_parities, n_data))
    message_d2p = np.ones((n_data, n_parities))

    # Initialization for all neighboring elements
    q_d2p = np.zeros((2, n_data, n_parities))
    for m, n in zip(*edges_p2d):
        q_d2p[0][n, m] = init_proba['I'][n] + init_proba[gf4_to_pauli(H[m, n])][n]
    q_d2p[1][edges_d2p] = 1 - q_d2p[0][edges_d2p]
    message_d2p[edges_d2p] = q_d2p[0][edges_d2p] - q_d2p[1][edges_d2p]

    for iter in range(max_iter):
        # -------- Parity to data -------

        for m, n in zip(*edges_p2d):
            message_p2d[m, n] = (-1)**syndrome[m] * np.prod(message_d2p[:, m])
            message_p2d[m, n] /= (message_d2p[n, m] + eps)

        # -------- Data to parity --------

        r_p2d = [(1 + message_p2d) / 2, (1 - message_p2d) / 2]
        for m, n in zip(*edges_p2d):
            q_d2p_pauli = {'I': init_proba['I'][n] * np.prod(r_p2d[0][:, n])}
            q_d2p_pauli['I'] /= (r_p2d[0][m, n] + eps)

            for pauli in paulis:
                q_d2p_pauli[pauli] = init_proba[pauli][n]

                for m_prime in range(n_parities):
                    if m_prime != m and H[m_prime, n] != 0:
                        innerp = commute(H[m_prime, n], pauli)
                        q_d2p_pauli[pauli] *= r_p2d[innerp][m_prime, n]

            q_d2p[0][n, m] = q_d2p_pauli['I'] + q_d2p_pauli[gf4_to_pauli(H[m, n])]
            q_d2p[1][n, m] = 0
            for pauli in paulis:
                if pauli != 'I' and pauli != H[m, n]:
                    q_d2p[1][n, m] += q_d2p_pauli[pauli]

            Z = q_d2p[0][n, m] + q_d2p[1][n, m]
            q_d2p[0][n, m] /= Z
            q_d2p[1][n, m] /= Z

            message_d2p[n, m] = q_d2p[0][n, m] - q_d2p[1][n, m]

        # -------- Soft decision --------

        q_pauli = {pauli: np.zeros(n_data) for pauli in paulis}
        for n in range(n_data):
            for pauli in paulis:
                q_pauli[pauli][n] = init_proba[pauli][n]
            for m in range(n_parities):
                if H[m, n] != 0:
                    q_pauli['I'][n] *= r_p2d[0][m, n]
                    for pauli in paulis:
                        innerp = commute(H[m, n], pauli)
                        q_pauli[pauli][n] *= r_p2d[innerp][m, n]

    predicted_probas = q_pauli
    correction = 0

    return correction, predicted_probas


def bp_osd_decoder(
    H: GF4, syndrome: np.ndarray, p: Dict[str, np.ndarray], max_bp_iter=10
) -> np.ndarray:
    
    print("Start BP decoding")
    correction, bp_probas = bp_decoder(H, syndrome, p, max_bp_iter)
    print("Predicted probas", bp_probas)
    correction = osd_decoder(H, syndrome, bp_probas)

    return correction


class BeliefPropagationOSDDecoder(BaseDecoder):
    label = 'BP-OSD decoder'

    def __init__(self, error_model: BaseErrorModel,
                 probability: float,
                 max_bp_iter: int = 10,
                 joschka: bool = True):
        super().__init__()
        self._error_model = error_model
        self._probability = probability
        self._max_bp_iter = max_bp_iter
        self._joschka = joschka

    def get_probabilities(
        self, code: StabilizerCode
    ) -> Tuple[np.ndarray, np.ndarray]:

        pi, px, py, pz = self._error_model.probability_distribution(code, self._probability)

        return pi, px, py, pz

    def update_probabilities(self, correction: np.ndarray,
                             px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                             direction: str = "x->z") -> np.ndarray:
        """Update X probabilities once a Z correction has been applied"""

        n_qubits = len(correction)

        new_probs = np.zeros(n_qubits)

        if direction == "z->x":
            for i in range(n_qubits):
                if correction[i] == 1:
                    if pz[i] + py[i] != 0:
                        new_probs[i] = py[i] / (pz[i] + py[i])
                else:
                    new_probs[i] = px[i] / (1 - pz[i] - py[i])

        elif direction == "x->z":
            for i in range(n_qubits):
                if correction[i] == 1:
                    if px[i] + py[i] != 0:
                        new_probs[i] = py[i] / (px[i] + py[i])
                else:
                    new_probs[i] = pz[i] / (1 - px[i] - py[i])

        else:
            raise ValueError(f"Unrecognized direction {direction} when updating probabilities")

        return new_probs

    # @profile
    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        if hasattr(code, 'gf_H'):
            H = code.gf_H
        else:
            raise NotImplementedError

        syndrome = np.array(syndrome, dtype=int)

        pi, px, py, pz = self.get_probabilities(code)
        probabilities = {'I': pi, 'X': px, 'Y': py, 'Z': pz}

        correction = bp_osd_decoder(
            H, syndrome, probabilities, max_bp_iter=self._max_bp_iter
        )

        correction = correction.astype(int)

        return correction


if __name__ == "__main__":
    from panqec.codes import RotatedPlanar3DCode
    import qecsim.paulitools as pt
    from panqec.error_models import PauliErrorModel

    np.random.seed(42)

    L = 1
    code = RotatedPlanar3DCode(L, L, L)
    n_qubits = code.n

    probability = 0.1
    r_x, r_y, r_z = [0.1, 0.1, 0.8]

    error_model = PauliErrorModel(r_x, r_y, r_z)
    errors = error_model.generate(code, probability)
    syndrome = pt.bsp(errors, code.stabilizer_matrix.T)

    H = code.gf_H
    pi, px, py, pz = error_model.probability_distribution(code, probability)
    probabilities = {'I': pi, 'X': px, 'Y': py, 'Z': pz}

    correction, bp_probas = bp_decoder(H, syndrome, probabilities)
    # decoder = BeliefPropagationOSDDecoder(error_model, probability, joschka=True)

    # correction = decoder.decode(code, syndrome)
