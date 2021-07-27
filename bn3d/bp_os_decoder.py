import numpy as np
from qecsim.model import Decoder, StabilizerCode
from typing import Tuple
import numpy.ma as ma


# @profile
def get_rref_mod2(A: np.ndarray, b: np.ndarray) -> Tuple(np.ndarray, np.ndarray):
    """Take a matrix A and a vector b.
    Return the row echelon form of A and a new vector b,
    modified with the same row operations"""
    n_rows, n_cols = A.shape
    A = A.copy()
    b = b.copy()

    i_pivot = 0
    i_col = 0
    while i_pivot < n_rows and i_col < n_cols:
        i_nonzero_row = np.argmax(A[i_pivot:, i_col]) + i_pivot

        if A[i_nonzero_row, i_col]:
            A[[i_pivot, i_nonzero_row]] = A[[i_nonzero_row, i_pivot]]
            b[[i_pivot, i_nonzero_row]] = b[[i_nonzero_row, i_pivot]]

            list_indices = np.where(A[:, i_col] == 1)[0]
            list_indices = np.delete(list_indices, np.where(list_indices == i_pivot))

            A[list_indices] += A[i_pivot]
            A[list_indices] = np.where(A[list_indices] != 1, 0, 1)  # faster than modulo 2
            b[list_indices] = (b[list_indices] - b[i_pivot]) % 2

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


def select_independent_columns(A: np.ndarray) -> list:
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
def osd_decoder(H: np.ndarray, syndrome: np.ndarray, bp_proba: np.ndarray) -> np.ndarray:
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
def bp_decoder(H: np.ndarray, syndrome: np.ndarray, p=0.3, max_iter=10) -> np.ndarray:
    """Belief propagation decoder.
    It returns the probability for each qubit to have an error
    """

    n_parities, n_data = H.shape

    log_ratio_p = np.log((1-p) / p)

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
    message_d2p[edges_p2d[1], edges_p2d[0]] = log_ratio_p

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
        message_d2p[edges_p2d[1], edges_p2d[0]] = log_ratio_p + sum_messages_data[edges_p2d[1]] - message_p2d[edges_p2d]

    # Soft decision
    sum_messages = np.sum(message_p2d, axis=0)
    log_ratio_error = log_ratio_p + sum_messages
    predicted_probas = 1 / (np.exp(log_ratio_error)+1)

    return predicted_probas


def bp_osd_decoder(H: np.ndarray, syndrome: np.ndarray, p=0.3, max_bp_iter=10) -> np.ndarray:
    bp_probas = bp_decoder(H, syndrome, p, max_bp_iter)
    correction = osd_decoder(H, syndrome, bp_probas)

    return correction


class BeliefPropagationOSDDecoder(Decoder):
    label = 'Toric 2D Belief Propagation + OSD decoder'

    def __init__(self):
        pass

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        n_vertices = int(np.product(code.size))
        n_qubits = code.n_k_d[0]

        n_stabilizers = code.stabilizers.shape[0]
        n_faces = n_stabilizers - n_vertices

        Hz = code.stabilizers[:n_faces, :n_qubits]
        Hx = code.stabilizers[n_faces:, n_qubits:]

        syndrome = np.array(syndrome, dtype=int)
        syndrome_z = syndrome[:n_faces]
        syndrome_x = syndrome[n_faces:]

        # H_z = code.stabilizers[:n_vertices, n_qubits:]
        # H_x = code.stabilizers[n_vertices:, :n_qubits]
        # syndrome_z = syndrome[:n_vertices]
        # syndrome_x = syndrome[n_vertices:]

        x_correction = bp_osd_decoder(Hx, syndrome_x, p=0.05, max_bp_iter=10)
        z_correction = bp_osd_decoder(Hz, syndrome_z, p=0.05, max_bp_iter=10)

        correction = np.concatenate([x_correction, z_correction])
        correction = correction.astype(int)

        return correction


if __name__ == "__main__":
    from bn3d.tc3d import ToricCode3D

    L = 9
    code = ToricCode3D(L, L, L)
    decoder = BeliefPropagationOSDDecoder()

    n_stabilizers = code.stabilizers.shape[0]
    syndrome = np.zeros(n_stabilizers)

    correction = decoder.decode(code, syndrome)
    # print(correction)
