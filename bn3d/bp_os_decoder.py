import numpy as np
from qecsim.model import Decoder, StabilizerCode
from flint import nmod_mat
from typing import Tuple, Dict
# from bn3d.array_ops import to_array, rref_mod2


def get_rref_mod2(A, syndrome):
    n_rows, n_cols = A.shape
    A = A.copy()

    i_pivot = 0
    i_col = 0
    while i_pivot < n_rows and i_col < n_cols:
        i_nonzero_row = i_pivot
        for i_row in range(i_pivot, n_rows):
            if A[i_row, i_col]:
                i_nonzero_row = i_row
                break

        if A[i_nonzero_row, i_col]:
            A[[i_pivot, i_nonzero_row]] = A[[i_nonzero_row, i_pivot]]
            syndrome[i_row] = (syndrome[i_row] - syndrome[i_pivot]) % 2
            
            for i_row in range(n_rows):
                if i_row != i_pivot and A[i_row, i_col]:
                    A[i_row] = (A[i_row] - A[i_pivot]) % 2
                    syndrome[i_row] = (syndrome[i_row] - syndrome[i_pivot]) % 2
                    
            i_pivot += 1
        i_col += 1

    return A, syndrome


def modular_rank(matrix):
    matrix = np.array(matrix, dtype=int)
    matrix_nmod = nmod_mat(matrix.shape[0], matrix.shape[1], matrix.ravel().tolist(), 2)

    return matrix_nmod.rank()


def modular_solve(A, b):
    n_rows, n_cols = A.shape

    # assert (np.diag(A) == np.ones(len(np.diag(A)))).all()
    A_nmod = nmod_mat(A.shape[0], A.shape[1], A.ravel().tolist(), 2)
    b_nmod = nmod_mat(len(b), 1, b.ravel().tolist(), 2)

    x_nmod = A_nmod.solve(b_nmod)

    x = np.array(x_nmod.table(), dtype=int)

    return x.ravel()


def select_independent_columns(A):
    n_rows, n_cols = A.shape
    # A_nmod = nmod_mat(n_rows, n_cols, A.ravel().tolist(), 2)
    # A_rref = nmod_mat.rref(A_nmod)[0]
    # A_rref = rref_mod2(A, syndrome)

    # print("Rank", modular_rank(A))
    # print("Full matrix\n", A)
    # print("Own rref\n", A_rref_2)
    # print("Their rref\n", np.array(A_rref.table()))
    # list_col_idx = select_columns_from_rref(A_rref)
    i_col, i_row = 0, 0
    list_col_idx = []
    while i_col < n_cols and i_row < n_rows:
        if A[i_row, i_col]:
            list_col_idx.append(i_col)
            i_row += 1
        i_col += 1

    # list_col_idx_2 = np.sort(np.unique(np.argmax(np.flip(A, axis=0), axis=0), return_index=True)[1])

    # A_rref_np = np.array(A_rref.table(), dtype=int)
    # print("A rref\n", A_rref_np[:, list_col_idx])

    return list_col_idx


def osd_decoder(H, syndrome, bp_proba):
    n_stabilizers, n_data = H.shape

    sorted_data_indices = list(np.argsort(-bp_proba))
    H_sorted = H[:, sorted_data_indices]

    # A = np.array(H_sorted, dtype=np.float64)
    H_sorted, syndrome = get_rref_mod2(H_sorted, syndrome)
    # print(H_rref)

    selected_col_indices = select_independent_columns(H_sorted)
    selected_row_indices = list(range(len(selected_col_indices)))
    # selected_row_indices = select_independent_columns(H_sorted[:, selected_col_indices].T)

    # print("H_rref\n", H_sorted[:, selected_col_indices])
    # aa

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

    def new_row_echelon_form(
        self, Hz: np.ndarray, Hx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        list_rref = []
        for H in [Hz, Hx]:
            H_nmod = nmod_mat(H.shape[0], H.shape[1], H.ravel().tolist(), 2)
            H_rref = to_array(nmod_mat.table(nmod_mat.rref(H_nmod)[0]))
            list_rref.append(H_rref)

        return tuple(list_rref)

    def get_row_echelon_form(
        self, code: StabilizerCode, Hz: np.ndarray, Hx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if code.label not in self._Hx_rref:
            rref = self.new_row_echelon_form(Hz, Hx)
            self._Hz_rref[code.label] = rref[0]
            self._Hx_rref[code.label] = rref[1]
        
        return (self._Hz_rref[code.label], self._Hx_rref[code.label])

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        n_vertices = int(np.product(code.size))
        n_qubits = code.n_k_d[0]

        n_stabilizers = code.stabilizers.shape[0]
        n_faces = n_stabilizers - n_vertices

        Hz = code.stabilizers[:n_faces, :n_qubits]
        Hx = code.stabilizers[n_faces:, n_qubits:]

        # Hz_rref, Hx_rref = self.get_row_echelon_form(code, Hz, Hx)

        syndrome = np.array(syndrome, dtype=int)
        syndrome_z = syndrome[:n_faces]
        syndrome_x = syndrome[n_faces:]

        # H_z = code.stabilizers[:n_vertices, n_qubits:]
        # H_x = code.stabilizers[n_vertices:, :n_qubits]
        # syndrome_z = syndrome[:n_vertices]
        # syndrome_x = syndrome[n_vertices:]

        x_correction = bp_osd_decoder(Hx, syndrome_x, p=0.1, max_bp_iter=10)
        z_correction = bp_osd_decoder(Hz, syndrome_z, p=0.1, max_bp_iter=10)

        print("Correction")
        print(x_correction)
        print(z_correction)
        correction = np.concatenate([x_correction, z_correction])
        correction = correction.astype(int)

        # print("Correction\n", correction)

        return correction


if __name__ == "__main__":
    from bn3d.tc3d import ToricCode3D

    # print(to_array([[0 for j in range(10)] for _ in range(1000)]))
    L = 9
    code = ToricCode3D(L, L, L)
    decoder = BeliefPropagationOSDDecoder()

    n_stabilizers = code.stabilizers.shape[0]
    syndrome = np.zeros(n_stabilizers)

    correction = decoder.decode(code, syndrome)
    print(correction)
