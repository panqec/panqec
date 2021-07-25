import numpy as np
from qecsim.model import Decoder, StabilizerCode
from typing import Dict
import numpy.ma as ma

# from bn3d.array_ops import rref_mod2


# @profile
def get_rref_mod2(A, b):
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


def solve_rref(A, b):
    """Solve the system Ax=b mod 2, with A in row echelon form"""
    n_rows, n_cols = A.shape
    x = np.zeros(n_rows)

    for i in range(n_rows-1, -1, -1):
        x[i] = b[i] - A[i].dot(x)

    return x % 2


def select_independent_columns(A):
    n_rows, n_cols = A.shape

    i_col, i_row = 0, 0
    list_col_idx = []
    while i_col < n_cols and i_row < n_rows:
        if A[i_row, i_col]:
            list_col_idx.append(i_col)
            i_row += 1
        i_col += 1

    return list_col_idx


def osd_decoder(H, syndrome, bp_proba):
    n_stabilizers, n_data = H.shape

    sorted_data_indices = list(np.argsort(-bp_proba))
    H_sorted = H[:, sorted_data_indices]

    H_sorted_rref, syndrome_rref = get_rref_mod2(H_sorted, syndrome)

    selected_col_indices = select_independent_columns(H_sorted_rref)
    selected_row_indices = list(range(len(selected_col_indices)))
    # selected_row_indices = select_independent_columns(H_sorted[:, selected_col_indices].T)

    reduced_H_rref = H_sorted_rref[selected_row_indices][:, selected_col_indices]
    reduced_syndrome_rref = syndrome_rref[selected_row_indices]

    reduced_correction = solve_rref(reduced_H_rref, reduced_syndrome_rref)
    # reduced_correction = modular_solve(reduced_H, reduced_syndrome)

    sorted_correction = np.zeros(n_data)
    for i, idx in enumerate(selected_col_indices):
        sorted_correction[idx] = reduced_correction[i]

    correction = np.zeros(n_data)
    for i, idx in enumerate(sorted_data_indices):
        correction[idx] = sorted_correction[i]

    return correction


# @profile
def bp_decoder(H, syndrome, p=0.3, max_iter=10):
    n_stabilizers, n_data = H.shape
    
    log_ratio_p = np.log((1-p) / p)

    edges_p2d = np.nonzero(H)
    message_d2p = ma.masked_array(np.inf * np.ones((n_data, n_stabilizers)), 1 - H.T)
    message_p2d = ma.masked_array(np.zeros((n_stabilizers, n_data)), 1 - H)

    # Initialization
    message_d2p[edges_p2d[1], edges_p2d[0]] = log_ratio_p

    for iter in range(max_iter):
        # Scaling factor
        alpha = 1 - 2**(-iter-1)

        # -------- Parity to data -------

        # Calculate sign neighboring messages
        prod_sign_parity = np.sign(np.prod(message_d2p, axis=0))
        sign_edges = np.sign(message_d2p[edges_p2d[1], edges_p2d[0]])
        prod_sign_neighbors = prod_sign_parity[edges_p2d[0]] * sign_edges

        # Calculate min neighboring messages
        abs_message_d2p = np.abs(message_d2p)
        argmin_abs_parity = np.argmin(abs_message_d2p, axis=0)
        min_abs_parity = abs_message_d2p[argmin_abs_parity, list(range(abs_message_d2p.shape[1]))]

        mask = np.ones((n_data, n_stabilizers), dtype=bool)
        mask[argmin_abs_parity, range(n_stabilizers)] = False
        new_abs_message_d2p = ma.masked_array(abs_message_d2p, ~mask)
        second_min_abs_parity = np.min(new_abs_message_d2p, axis=0)

        abs_edges = np.abs(message_d2p[edges_p2d[1], edges_p2d[0]])
        cond = abs_edges > min_abs_parity[edges_p2d[0]]
        min_neighbors = np.select([cond, ~cond], [min_abs_parity[edges_p2d[0]], second_min_abs_parity[edges_p2d[0]]])

        # Update the message
        message_p2d[edges_p2d] = -(2*syndrome[edges_p2d[0]]-1) * alpha
        message_p2d[edges_p2d] *= prod_sign_neighbors
        message_p2d[edges_p2d] *= min_neighbors

        # -------- Data to parity --------
        sum_messages_data = np.sum(message_p2d, axis=0)
        message_d2p[edges_p2d[1], edges_p2d[0]] = log_ratio_p + sum_messages_data[edges_p2d[1]] - message_p2d[edges_p2d]

    # Hard decision
    sum_messages = np.sum(message_p2d, axis=0)
    log_ratio_error = log_ratio_p + sum_messages

    predicted_probas = 1 / (np.exp(log_ratio_error)+1)

    return correction, predicted_probas


def bp_osd_decoder(H, syndrome, p=0.3, max_bp_iter=10):
    bp_correction, bp_probas = bp_decoder(H, syndrome, p, max_bp_iter)
    correction = osd_decoder(H, syndrome, bp_probas)

    return correction


class BeliefPropagationOSDDecoder(Decoder):
    label = 'Toric 2D Belief Propagation + OSD decoder'

    _Hx_rref: Dict[str, np.ndarray] = {}
    _Hz_rref: Dict[str, np.ndarray] = {}

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
