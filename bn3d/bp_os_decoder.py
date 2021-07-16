import numpy as np
from qecsim.model import Decoder, StabilizerCode
from flint import nmod_mat
from scipy.linalg import lu
from scipy.sparse import csr_matrix
import networkx as nx


def get_list_edges(adjacency):
    edges = []
    for i_data in range(len(adjacency)):
        for i_parity in range(len(adjacency[i_data])):
            edges.append((i_data, adjacency[i_data][i_parity]))

    return edges


def projection(vector, list_vectors):
    vector = vector.reshape(len(vector), 1)
    return np.sum(list_vectors * vector.T.dot(list_vectors) / np.linalg.norm(list_vectors, axis=0)**2, axis=1)


def select_independent_columns(A, rank):
    print("Test", np.any(A))
    first_idx = np.where(A.any(0))[0][0]

    list_col_idx = [first_idx]
    for col in range(first_idx + 1, A.shape[1]):
        if modular_rank(A[:, list_col_idx + [col]]) > len(list_col_idx):
            list_col_idx.append(col)
        if len(list_col_idx) == rank:
            break

    return list_col_idx


def modular_solve(A, b):
    n_rows, n_cols = A.shape

    A = np.array(A, dtype=int)[:n_cols]
    b = np.array(b, dtype=int)[:n_cols]

    # assert (np.diag(A) == np.ones(len(np.diag(A)))).all()
    A_nmod = nmod_mat(A.shape[0], A.shape[1], A.ravel().tolist(), 2)
    b_nmod = nmod_mat(len(b), 1, b.ravel().tolist(), 2)

    x_nmod = A_nmod.solve(b_nmod)

    x = np.array(x_nmod.table(), dtype=int)

    return x.ravel()


def modular_rank(matrix):
    matrix = np.array(matrix, dtype=int)
    matrix_nmod = nmod_mat(matrix.shape[0], matrix.shape[1], matrix.ravel().tolist(), 2)
   
    return matrix_nmod.rank()


def bp_decoder(H, syndrome, p=0.3, max_iter=10):
    n_stabilizers, n_data = H.shape

    neighbor_parity = [np.nonzero(H[i])[0].tolist() for i in range(n_stabilizers)]
    neighbor_data = [np.nonzero(H[:, j])[0].tolist() for j in range(n_data)]

    edges = []
    for i in range(n_data):
        for j in neighbor_data[i]:
            edges.append((i, j))

    log_ratio_p = np.log((1-p) / p)

    message_data_to_parity = {edge: 0 for edge in edges}
    message_parity_to_data = {edge[::-1]: 1 for edge in edges}

    # Initialization
    for edge in edges:
        message_data_to_parity[edge] = log_ratio_p

    log_ratio_error = [0 for _ in range(n_data)]
    correction = [0 for _ in range(n_data)]
    predicted_probas = np.zeros(n_data)

    for iter in range(max_iter):
        # Scaling factor
        alpha = 1 - 2**(-iter-1)

        # Parity to data
        for edge in edges:
            min_messages = np.inf
            prod_sign_messages = 1
            
            for i_data in neighbor_parity[edge[1]]:  
                if i_data != edge[0]:
                    message = message_data_to_parity[(i_data, edge[1])]
                    min_messages = min(min_messages, np.abs(message))
                    prod_sign_messages = prod_sign_messages * np.sign(message)

            message_parity_to_data[(edge[1], edge[0])] = -(2*syndrome[edge[1]]-1) * alpha * prod_sign_messages * min_messages

        # Data to parity
        for edge in edges:
            sum_messages = log_ratio_p
            for i_parity in neighbor_data[edge[0]]:
                if i_parity != edge[1]:
                    sum_messages += message_parity_to_data[(i_parity, edge[0])]
                    
            message_data_to_parity[edge[0], edge[1]] = sum_messages

        # Hard decision
        for i_data in range(n_data):
            log_ratio_error[i_data] = log_ratio_p + np.sum([message_parity_to_data[(i_parity, i_data)]
                                                            for i_parity in neighbor_data[i_data]])
        
        correction = np.where(np.array(log_ratio_error) <= 0)[0]
        predicted_probas = 1 / (np.exp(log_ratio_error)+1)

    return correction, predicted_probas


def osd_decoder(H, syndrome, bp_proba):
    # print("H", H)
    n_stabilizers, n_data = H.shape
    syndrome = np.array(syndrome, dtype=np.uint)

    rank = modular_rank(H)
    sorted_data_indices = list(np.argsort(-bp_proba))
    H_sorted = H[:, sorted_data_indices]

    A = np.array(H_sorted, dtype=np.float64)
    selected_indices = select_independent_columns(A, rank)

    reduced_H = H_sorted[:, selected_indices]

    assert np.linalg.matrix_rank(reduced_H) == rank

    P, L, U = lu(reduced_H)

    L = np.array(np.mod(L, 2), dtype=int)
    U = np.array(np.mod(U, 2), dtype=int)

    Pinv = np.linalg.inv(P) % 2

    y = modular_solve(L, Pinv.dot(syndrome))
    reduced_correction = modular_solve(U, y)

    sorted_correction = np.zeros(n_data)
    for i, idx in enumerate(selected_indices):
        sorted_correction[idx] = reduced_correction[i]

    correction = np.zeros(n_data)
    for i, idx in enumerate(sorted_data_indices):
        correction[idx] = sorted_correction[i]

    return correction


def bp_osd_decoder(H, syndrome, p=0.3, max_bp_iter=10):
    bp_correction, bp_probas = bp_decoder(H, syndrome, p, max_bp_iter)
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

        H_z = code.stabilizers[:n_vertices, n_qubits:]
        H_x = code.stabilizers[n_vertices:, :n_qubits]

        syndrome_z = syndrome[:n_vertices]
        syndrome_x = syndrome[n_vertices:]

        z_correction = bp_osd_decoder(H_z, syndrome_z, p=0.1, max_bp_iter=10)
        x_correction = bp_osd_decoder(H_x, syndrome_x, p=0.1, max_bp_iter=10)

        correction = np.concatenate([z_correction, x_correction])
        correction = correction.astype(np.int)

        return correction
