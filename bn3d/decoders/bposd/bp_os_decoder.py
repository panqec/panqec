from typing import Dict
import numpy as np
from qecsim.model import Decoder, StabilizerCode, ErrorModel
from typing import Tuple, List
import numpy.ma as ma
from ldpc import bposd_decoder


def get_rref_mod2(
    A: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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

    # Create a full-rank squared matrix, by selecting independent columns and
    # rows
    selected_col_indices = select_independent_columns(H_sorted_rref)
    selected_row_indices = list(range(len(selected_col_indices)))
    reduced_H_rref = H_sorted_rref[selected_row_indices][
        :, selected_col_indices
    ]
    reduced_syndrome_rref = syndrome_rref[selected_row_indices]

    # Solve the system H*e = s, in its rref
    reduced_correction = solve_rref(reduced_H_rref, reduced_syndrome_rref)

    # Fill with the 0 the non-selected (-> dependent columns) indices
    sorted_correction = np.zeros(n_data)
    sorted_correction[selected_col_indices] = reduced_correction

    # Rearrange the indices of the correction to take the initial sorting into
    # account.
    correction = np.zeros(n_data)
    correction[sorted_data_indices] = sorted_correction

    return correction


def bp_decoder(H: np.ndarray,
               syndrome: np.ndarray,
               probabilities: np.ndarray,
               max_iter=10,
               eps=1e-8) -> Tuple[np.ndarray, np.ndarray]:
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
    # The initialization with np.inf (resp. zero) allows to ignore
    # non-neighboring elements when doing a min (resp. sum)
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

        # For each edge, calculate sign of the neighbors of the parity bit in
        # that edge excluding the edge itself
        prod_sign_neighbors = prod_sign_parity[edges_p2d[0]] * sign_edges

        # Calculate minimum of the neighboring messages (in absolute value) for
        # each edge excluding that edge itself.
        # For that calculate the absolute value of each message
        abs_message_d2p = np.abs(message_d2p)

        # Then calculate the min and second min of the neighbors at each parity
        # bit.
        argmin_abs_parity = np.argmin(abs_message_d2p, axis=0)
        min_abs_parity = abs_message_d2p[
            argmin_abs_parity, list(range(abs_message_d2p.shape[1]))
        ]
        mask = np.ones((n_data, n_parities), dtype=bool)
        mask[argmin_abs_parity, range(n_parities)] = False
        new_abs_message_d2p: ma.MaskedArray = ma.masked_array(
            abs_message_d2p, ~mask
        )
        second_min_abs_parity = np.min(new_abs_message_d2p, axis=0)

        # It allows to calculate the minimum excluding the edge
        abs_edges = np.abs(message_d2p[edges_p2d[1], edges_p2d[0]])
        cond = abs_edges > min_abs_parity[edges_p2d[0]]
        min_neighbors = np.select(
            [cond, ~cond],
            [min_abs_parity[edges_p2d[0]], second_min_abs_parity[edges_p2d[0]]]
        )

        # Update the message
        message_p2d[edges_p2d] = -(2*syndrome[edges_p2d[0]]-1) * alpha
        message_p2d[edges_p2d] *= prod_sign_neighbors
        message_p2d[edges_p2d] *= min_neighbors

        # -------- Data to parity --------

        # Sum messages at each data bit
        sum_messages_data = np.sum(message_p2d, axis=0) + eps

        # For each edge, get the sum around the data bit, excluding that edge
        message_d2p[edges_p2d[1], edges_p2d[0]] = (
            log_ratio_p[edges_p2d[1]]
            + sum_messages_data[edges_p2d[1]] - message_p2d[edges_p2d]
        )

        # Soft decision
        sum_messages = np.sum(message_p2d, axis=0)
        log_ratio_error = log_ratio_p + sum_messages
        correction = (log_ratio_error < 0).astype(np.uint)

        if np.all(np.mod(H.dot(correction), 2) == syndrome):
            break

    predicted_probas = 1 / (np.exp(log_ratio_error)+1)

    return correction, predicted_probas


def bp_osd_decoder(
    H: np.ndarray, syndrome: np.ndarray, p=0.3, max_bp_iter=10
) -> np.ndarray:
    correction, bp_probas = bp_decoder(H, syndrome, p, max_bp_iter)
    if np.any(np.mod(H.dot(correction), 2) != syndrome):
        correction = osd_decoder(H, syndrome, bp_probas)

    return correction


class BeliefPropagationOSDDecoder(Decoder):
    label = 'BP-OSD decoder'

    def __init__(self, error_model: ErrorModel,
                 probability: float,
                 max_bp_iter: int = 10,
                 joschka: bool = True,
                 channel_update: bool = False):
        super().__init__()
        self._error_model = error_model
        self._probability = probability
        self._max_bp_iter = max_bp_iter
        self._joschka = joschka
        self._channel_update = channel_update

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

    def update_probabilities(self, correction: np.ndarray,
                             px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                             direction: str = "x->z") -> np.ndarray:
        """Update X probabilities once a Z correction has been applied"""

        n_qubits = correction.shape[0]

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
            raise ValueError(
                f"Unrecognized direction {direction} when "
                "updating probabilities"
            )

        return new_probs

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        is_css = code.is_css

        n_qubits = code.n
        syndrome = np.array(syndrome, dtype=int)

        if is_css:
            syndrome_z = syndrome[code.Hz.shape[0]:]
            syndrome_x = syndrome[:code.Hz.shape[0]]

        pi, px, py, pz = self.get_probabilities(code)

        probabilities_x = px + py
        probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if self._joschka:
            # Load saved decoders
            if code.label in self._x_decoder.keys():
                x_decoder = self._x_decoder[code.label]
                z_decoder = self._z_decoder[code.label]
            elif code.label in self._decoder.keys():
                decoder = self._decoder[code.label]

            # Initialize new decoders otherwise
            else:
                if is_css:
                    z_decoder = bposd_decoder(
                        code.Hx,
                        error_rate=0.05,  # ignore this due to the next parameter
                        channel_probs=probabilities_z,
                        max_iter=self._max_bp_iter,
                        bp_method="msl",
                        ms_scaling_factor=0,
                        osd_method="osd_cs",  # Choose from: "osd_e", "osd_cs", "osd0"
                        osd_order=6
                    )

                    x_decoder = bposd_decoder(
                        code.Hz,
                        error_rate=0.05,  # ignore this due to the next parameter
                        channel_probs=probabilities_x,
                        max_iter=self._max_bp_iter,
                        bp_method="msl",
                        ms_scaling_factor=0,
                        osd_method="osd_cs",  # Choose from: "osd_e", "osd_cs", "osd0"
                        osd_order=6
                    )
                    self._x_decoder[code.label] = x_decoder
                    self._z_decoder[code.label] = z_decoder
                else:
                    decoder = bposd_decoder(
                        code.stabilizer_matrix,
                        error_rate=0.05,  # ignore this due to the next parameter,
                        channel_probs=probabilities,
                        max_iter=self._max_bp_iter,
                        bp_method="msl",
                        ms_scaling_factor=0,
                        osd_method="osd_cs",  # Choose from: "osd_e", "osd_cs", "osd0"
                        osd_order=6
                    )
                    self._decoder[code.label] = decoder

            if is_css:
                # Update probabilities (in case the distribution is new at each iteration)
                x_decoder.update_channel_probs(probabilities_x)
                z_decoder.update_channel_probs(probabilities_z)

                # Decode Z errors
                z_decoder.decode(syndrome_z)
                z_correction = z_decoder.osdw_decoding

                # Bayes update of the probability
                if self._channel_update:
                    new_x_probs = self.update_probabilities(
                        z_correction, px, py, pz, direction="z->x"
                    )
                    x_decoder.update_channel_probs(new_x_probs)

                # Decode X errors
                x_decoder.decode(syndrome_x)
                x_correction = x_decoder.osdw_decoding

                correction = np.concatenate([x_correction, z_correction])
            else:
                # Update probabilities (in case the distribution is new at each iteration)
                decoder.update_channel_probs(probabilities)

                # Decode all errors
                decoder.decode(syndrome)
                correction = decoder.osdw_decoding
                correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])
        else:
            if is_css:
                z_correction = bp_osd_decoder(
                    code.Hx, syndrome_z, probabilities_z, max_bp_iter=self._max_bp_iter
                )
                if self._channel_update:
                    new_x_probs = self.update_probabilities(z_correction, px, py, pz)
                else:
                    new_x_probs = probabilities_x
                x_correction = bp_osd_decoder(
                    code.Hz, syndrome_x, new_x_probs, max_bp_iter=self._max_bp_iter
                )
                correction = np.concatenate([x_correction, z_correction])
            else:
                correction = bp_osd_decoder(
                    code.stabilizer_matrix, syndrome, probabilities, max_bp_iter=self._max_bp_iter
                )
                correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])

            correction = correction.astype(int)

        return correction


def test_decoder():
    from bn3d.models import Toric3DCode
    from bn3d.bpauli import bcommute, get_effective_error
    # from bn3d.noise import PauliErrorModel
    from bn3d.error_models import DeformedRandomErrorModel
    import time
    rng = np.random.default_rng()

    L = 20
    code = Toric3DCode(L, L, L)

    probability = 0.1
    r_x, r_y, r_z = [0.1, 0.1, 0.8]
    error_model = DeformedRandomErrorModel(r_x, r_y, r_z, p_xz=0.5, p_yz=0.5)

    decoder = BeliefPropagationOSDDecoder(
        error_model, probability, joschka=True
    )

    # Start timer
    start = time.time()

    n_iter = 5
    for i in range(n_iter):
        print(f"\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, probability=probability, rng=rng)
        print("Calculate syndrome")
        syndrome = bcommute(code.stabilizer_matrix, error)
        print("Decode")
        correction = decoder.decode(code, syndrome)
        print("Get total error")
        total_error = (correction + error) % 2
        print("Get effective error")
        effective_error = get_effective_error(
            total_error, code.logicals_x, code.logicals_z
        )
        print("Check codespace")
        codespace = bool(np.all(bcommute(code.stabilizer_matrix, total_error) == 0))
        success = bool(np.all(effective_error == 0)) and codespace
        print(success)

    print("Average time per iteration", (time.time() - start) / n_iter)


if __name__ == '__main__':
    test_decoder()
