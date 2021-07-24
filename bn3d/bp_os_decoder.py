import numpy as np
from qecsim.model import Decoder, StabilizerCode
from flint import nmod_mat
from scipy.linalg import lu
from scipy.sparse import csr_matrix
# import networkx as nx


def get_rref_mod2(A):
    n_rows, n_cols = A.shape
    A = A.copy()

    i_pivot = 0
    i_col = 0
    while i_pivot < n_rows and i_col < n_cols:
        # print(i_col, i_pivot)
        # print(np.array(A), "\n")
        i_nonzero_row = i_pivot
        for i_row in range(i_pivot, n_rows):
            if A[i_row, i_col]:
                i_nonzero_row = i_row
                break

        if A[i_nonzero_row, i_col]:
            A[i_pivot], A[i_nonzero_row] = A[i_nonzero_row].copy(), A[i_pivot].copy()
            i_pivot += 1

        for i_row in range(i_pivot, n_rows):
            if A[i_row, i_col]:
                A[i_row] = (A[i_row] - A[i_col]) % 2

        i_col += 1

    return A


def get_list_edges(adjacency):
    edges = []
    for i_data in range(len(adjacency)):
        for i_parity in range(len(adjacency[i_data])):
            edges.append((i_data, adjacency[i_data][i_parity]))

    return edges


def projection(vector, list_vectors):
    vector = vector.reshape(len(vector), 1)
    return np.sum(list_vectors * vector.T.dot(list_vectors) / np.linalg.norm(list_vectors, axis=0)**2, axis=1)


def select_independent_columns(A):
    n_rows, n_cols = A.shape
    A_nmod = nmod_mat(n_rows, n_cols, A.ravel().tolist(), 2)
    A_rref = nmod_mat.rref(A_nmod)[0]
    # A_rref = [[int(A_rref[(i, j)]) for j in range(n_cols)] for i in range(n_rows)]
    A_rref = list(map(lambda x: list(map(int, x)), nmod_mat.table(A_rref)))
    # print(A_rref[(1,0)])
    # aaa

    # A_rref_2 = get_rref_mod2(A)

    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print(np.array(A), "\n")
    # print(np.array(A_rref, dtype=int), "\n")
    # print(np.array(A_rref_2, dtype=int))
    # print(np.all(A_rref == A_rref_2))
    # aaa

    # A_rref_sparse = csr_matrix(A_rref)

    # list_col_idx = []
    # last_row_one = -1
    # max_last_row_one = -1
    # for i_col in range(n_cols):
    #     for i_row in range(n_rows):
    #         if A_rref[(i_row, i_col)]:
    #             last_row_one = i_row

    #     if last_row_one > max_last_row_one:
    #         list_col_idx.append(i_col)

    #     max_last_row_one = max(last_row_one, max_last_row_one)

    list_col_idx = np.sort(np.unique(np.argmax(np.flip(A_rref, axis=0), axis=0), return_index=True)[1])
    # print(list_col_idx)
    # print(list_col_idx_2)
    # aa

    return list_col_idx


def modular_solve(A, b):
    n_rows, n_cols = A.shape

    # assert (np.diag(A) == np.ones(len(np.diag(A)))).all()
    A_nmod = nmod_mat(A.shape[0], A.shape[1], A.ravel().tolist(), 2)
    b_nmod = nmod_mat(len(b), 1, b.ravel().tolist(), 2)

    x_nmod = A_nmod.solve(b_nmod)

    x = np.array(x_nmod.table(), dtype=int)

    return x.ravel()


def modular_rank(matrix):
    # print("Matrix", matrix)
    matrix = np.array(matrix, dtype=int)
    matrix_nmod = nmod_mat(matrix.shape[0], matrix.shape[1], matrix.ravel().tolist(), 2)
    # print(matrix)
    # print("Row echelon\n", nmod_mat.rref(matrix_nmod))
    # print("Column echelon\n", nmod_mat.rref(matrix_nmod.transpose()))

    return matrix_nmod.rank()


def bp_decoder(H, syndrome, p=0.3, max_iter=10):
    n_stabilizers, n_data = H.shape

    neighbor_parity = [np.nonzero(H[i])[0].tolist()
                       for i in range(n_stabilizers)]
    neighbor_data = [np.nonzero(H[:, j])[0].tolist()
                     for j in range(n_data)]

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
            sum_messages = np.sum([message_parity_to_data[(i_parity, i_data)]
                                   for i_parity in neighbor_data[i_data]])
            log_ratio_error[i_data] = log_ratio_p + sum_messages

        correction = np.where(np.array(log_ratio_error) <= 0)[0]
        predicted_probas = 1 / (np.exp(log_ratio_error)+1)

    return correction, predicted_probas


def osd_decoder(H, syndrome, bp_proba):
    n_stabilizers, n_data = H.shape
    syndrome = np.array(syndrome, dtype=np.uint)

    sorted_data_indices = list(np.argsort(-bp_proba))
    H_sorted = H[:, sorted_data_indices]

    # A = np.array(H_sorted, dtype=np.float64)
    selected_col_indices = select_independent_columns(H_sorted)
    selected_row_indices = select_independent_columns(H_sorted[:, selected_col_indices].T)

    reduced_H = H_sorted[selected_row_indices][:, selected_col_indices]
    reduced_syndrome = syndrome[selected_row_indices]

    reduced_correction = modular_solve(reduced_H, reduced_syndrome)

    sorted_correction = np.zeros(n_data)
    for i, idx in enumerate(selected_col_indices):
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

        n_stabilizers = code.stabilizers.shape[0]
        n_faces = n_stabilizers - n_vertices

        H_z = code.stabilizers[:n_faces, :n_qubits]
        H_x = code.stabilizers[n_faces:, n_qubits:]

        syndrome_z = syndrome[:n_faces]
        syndrome_x = syndrome[n_faces:]

        # H_z = code.stabilizers[:n_vertices, n_qubits:]
        # H_x = code.stabilizers[n_vertices:, :n_qubits]

        # syndrome_z = syndrome[:n_vertices]
        # syndrome_x = syndrome[n_vertices:]

        # print("H_x\n", H_x)
        # print("H_z\n", H_z)
        # print("syndrome_x\n", syndrome_x)
        # print("syndrome_z\n", syndrome_z)

        x_correction = bp_osd_decoder(H_x, syndrome_x, p=0.1, max_bp_iter=10)
        z_correction = bp_osd_decoder(H_z, syndrome_z, p=0.1, max_bp_iter=10)

        correction = np.concatenate([x_correction, z_correction])
        correction = correction.astype(np.int)

        # print("Correction\n", correction)

        return correction


if __name__ == "__main__":
    from bn3d.tc3d import ToricCode3D

    L = 5
    code = ToricCode3D(L, L, L)
    decoder = BeliefPropagationOSDDecoder()

    n_stabilizers = code.stabilizers.shape[0]
    syndrome = np.zeros(n_stabilizers)

    decoder.decode(code, syndrome)
