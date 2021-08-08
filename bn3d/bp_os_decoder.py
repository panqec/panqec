import numpy as np
import itertools
from qecsim.model import Decoder, StabilizerCode, ErrorModel
from typing import Tuple, List
import numpy.ma as ma
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix, find


def get_rref_mod2(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Take a matrix A and a vector b.
    Return the row echelon form of A and a new vector b,
    modified with the same row operations"""
    n_rows, n_cols = A.shape
    A = A.copy()
    b = b.copy()
    # A_sparse = coo_matrix(A)

    i_pivot = 0
    i_col = 0
    while i_pivot < n_rows and i_col < n_cols:
        i_nonzero_row = np.argmax(A[i_pivot:, i_col]) + i_pivot

        if A[i_nonzero_row, i_col]:
            A[[i_pivot, i_nonzero_row]] = A[[i_nonzero_row, i_pivot]]
            b[[i_pivot, i_nonzero_row]] = b[[i_nonzero_row, i_pivot]]

            cond = A[:, i_col] == 1
            cond[i_pivot] = False

            A[cond] = np.logical_xor(A[cond], A[i_pivot])
            b[cond] = np.logical_xor(b[cond], b[i_pivot])

            i_pivot += 1
        i_col += 1

    return A, b


def solve_rref(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the system Ax=b mod 2, with A in reduced row echelon form"""
    n_rows, n_cols = A.shape
    x = np.zeros(n_rows)

    for i in range(n_rows-1, -1, -1):
        x[i] = b[i] - A[i].dot(x)

    return x % 2


def select_independent_columns(A: np.ndarray) -> List[int]:
    """Select independent columns of a matrix A in reduced row echelon form"""
    n_rows, n_cols = A.shape

    i_col, i_row = 0, 0
    list_col_idx = []
    while i_col < n_cols and i_row < n_rows:
        if A[i_row, i_col]:
            list_col_idx.append(i_col)
            i_row += 1
        i_col += 1

    return list_col_idx


# @profile
def osd_decoder(H: np.ndarray,
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
    H_sorted_rref, syndrome_rref = get_rref_mod2(H_sorted, syndrome)

    # Create a full-rank squared matrix, by selecting independent columns and rows
    selected_col_indices = select_independent_columns(H_sorted_rref)
    selected_row_indices = list(range(len(selected_col_indices)))
    reduced_H_rref = H_sorted_rref[selected_row_indices][:, selected_col_indices]
    reduced_syndrome_rref = syndrome_rref[selected_row_indices]

    # Solve the system H*e = s, in its rref
    reduced_correction = solve_rref(reduced_H_rref, reduced_syndrome_rref)

    # Fill with the 0 the non-selected (-> dependent columns) indices
    sorted_correction = np.zeros(n_data)
    sorted_correction[selected_col_indices] = reduced_correction

    # Rearrange the indices of the correction to take the initial sorting into account
    correction = np.zeros(n_data)
    correction[sorted_data_indices] = sorted_correction

    return correction


# @profile
def bp_decoder(H: np.ndarray,
               syndrome: np.ndarray,
               probabilities: np.ndarray,
               max_iter=10,
               eps=1e-8) -> np.ndarray:
    """Belief propagation decoder.
    It returns a probability for each qubit to have had an error
    """

    n_parities, n_data = H.shape

    log_ratio_p = np.log((1-probabilities-eps) / (probabilities+eps))

    # Create tuple with parity indices and data indices
    # Each element (edges_p2d[0, i], edges_p2d[1, i]) is an edge
    # from parity to data
    edges_p2d = np.nonzero(H)

    # Create messages from parity to data and from data to parity
    # The initialization with np.inf (resp. zero) allows to ignore non-neighboring
    # elements when doing a min (resp. sum)
    message_d2p = np.inf * np.ones((n_data, n_parities))
    message_p2d = np.zeros((n_parities, n_data))

    # Initialization for all neighboring elements
    message_d2p[edges_p2d[1], edges_p2d[0]] = log_ratio_p[edges_p2d[1]]

    for iter in range(max_iter):        
        # Scaling factor
        alpha = 1 - 2**(-iter-1)

        # -------- Parity to data -------

        # Calculate sign of neighboring messages for each parity bit
        prod_sign_parity = np.sign(np.prod(message_d2p, axis=0))

        # Calculate sign of each message
        sign_edges = np.sign(message_d2p[edges_p2d[1], edges_p2d[0]])

        # For each edge, calculate sign of the neighbors of the parity bit in that edge
        # excluding the edge itself
        prod_sign_neighbors = prod_sign_parity[edges_p2d[0]] * sign_edges

        # Calculate minimum of the neighboring messages (in absolute value) for each edge
        # excluding that edge itself.
        # For that calculate the absolute value of each message
        abs_message_d2p = np.abs(message_d2p)

        # Then calculate the min and second min of the neighbors at each parity bit
        argmin_abs_parity = np.argmin(abs_message_d2p, axis=0)
        min_abs_parity = abs_message_d2p[argmin_abs_parity, list(range(abs_message_d2p.shape[1]))]
        mask = np.ones((n_data, n_parities), dtype=bool)
        mask[argmin_abs_parity, range(n_parities)] = False
        new_abs_message_d2p = ma.masked_array(abs_message_d2p, ~mask)
        second_min_abs_parity = np.min(new_abs_message_d2p, axis=0)

        # It allows to calculate the minimum excluding the edge
        abs_edges = np.abs(message_d2p[edges_p2d[1], edges_p2d[0]])
        cond = abs_edges > min_abs_parity[edges_p2d[0]]
        min_neighbors = np.select([cond, ~cond], [min_abs_parity[edges_p2d[0]], second_min_abs_parity[edges_p2d[0]]])

        # Update the message
        message_p2d[edges_p2d] = -(2*syndrome[edges_p2d[0]]-1) * alpha
        message_p2d[edges_p2d] *= prod_sign_neighbors
        message_p2d[edges_p2d] *= min_neighbors

        # -------- Data to parity --------

        # Sum messages at each data bit
        sum_messages_data = np.sum(message_p2d, axis=0)

        # For each edge, get the sum around the data bit, excluding that edge
        message_d2p[edges_p2d[1], edges_p2d[0]] = log_ratio_p[edges_p2d[1]] + sum_messages_data[edges_p2d[1]] - message_p2d[edges_p2d]

        # Soft decision
        sum_messages = np.sum(message_p2d, axis=0)
        log_ratio_error = log_ratio_p + sum_messages
        correction = (log_ratio_error < 0).astype(np.uint)
        
        if np.all(H.dot(correction) % 2 == syndrome):
            break

    predicted_probas = 1 / (np.exp(log_ratio_error)+1)

    return correction, predicted_probas


def bp_osd_decoder(H: np.ndarray, syndrome: np.ndarray, p=0.3, max_bp_iter=10) -> np.ndarray:
    correction, bp_probas = bp_decoder(H, syndrome, p, max_bp_iter)
    if np.any(H.dot(correction) % 2 != syndrome):
        correction = osd_decoder(H, syndrome, bp_probas)

    return correction


class BeliefPropagationOSDDecoder(Decoder):
    label = 'Toric 2D Belief Propagation + OSD decoder'

    def __init__(self, error_model: ErrorModel,
                 probability: float,
                 max_bp_iter: int = 10,
                 deformed: bool = False):
        super().__init__()
        self._error_model = error_model
        self._probability = probability
        self._deformed = deformed
        self._max_bp_iter = max_bp_iter

    def get_probabilities(self, code: StabilizerCode) -> Tuple[np.ndarray, np.ndarray]:
        r_x, r_y, r_z = self._error_model.direction
        p_X, p_Y, p_Z = np.array([r_x, r_y, r_z])*self._probability

        p_regular_x = p_X + p_Y
        p_regular_z = p_Z + p_Y
        p_deformed_x = p_Z + p_Y
        p_deformed_z = p_X + p_Y

        deformed_edge = code.X_AXIS

        probabilities_x = np.ones(code.shape, dtype=float)*p_regular_x
        probabilities_z = np.ones(code.shape, dtype=float)*p_regular_z

        if self._deformed:
            # The weights on the deformed edge are different
            ranges = [range(length) for length in code.shape]
            for axis, x, y, z in itertools.product(*ranges):
                if axis == deformed_edge:
                    probabilities_x[axis, x, y, z] = p_deformed_x
                    probabilities_z[axis, x, y, z] = p_deformed_z
   
        return probabilities_x.flatten(), probabilities_z.flatten()

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        Hz = code.Hz
        Hx = code.Hx

        syndrome = np.array(syndrome, dtype=int)

        syndrome_z = syndrome[:len(Hz)]
        syndrome_x = syndrome[len(Hz):]

        # H_z = code.stabilizers[:n_vertices, n_qubits:]
        # H_x = code.stabilizers[n_vertices:, :n_qubits]
        # syndrome_z = syndrome[:n_vertices]
        # syndrome_x = syndrome[n_vertices:]

        probabilities_x, probabilities_z = self.get_probabilities(code)

        x_correction = bp_osd_decoder(Hx, syndrome_x, probabilities_x, max_bp_iter=self._max_bp_iter)
        z_correction = bp_osd_decoder(Hz, syndrome_z, probabilities_z, max_bp_iter=self._max_bp_iter)

        correction = np.concatenate([x_correction, z_correction])
        correction = correction.astype(int)

        return correction


if __name__ == "__main__":
    from bn3d.tc3d import ToricCode3D
    import qecsim.paulitools as pt
    from bn3d.noise import PauliErrorModel

    L = 9
    code = ToricCode3D(L, L, L)

    probability = 0.1
    r_x, r_y, r_z = [0.1, 0.1, 0.8]

    error_model = PauliErrorModel(r_x, r_y, r_z)
    errors = error_model.generate(code, probability)
    syndrome = pt.bsp(errors, code.stabilizers.T)

    decoder = BeliefPropagationOSDDecoder(error_model, probability)

    correction = decoder.decode(code, syndrome)
