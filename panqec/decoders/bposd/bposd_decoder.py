import numpy as np
from typing import Tuple
from ldpc import bposd_decoder
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel
from panqec.decoders import BaseDecoder


class BeliefPropagationOSDDecoder(BaseDecoder):
    label = 'BP-OSD decoder'

    def __init__(self,
                 code: StabilizerCode,
                 error_model: BaseErrorModel,
                 max_bp_iter: int = 1000,
                 channel_update: bool = False,
                 osd_order: int = 10):
        super().__init__(code, error_model)
        self._max_bp_iter = max_bp_iter
        self._channel_update = channel_update
        self._osd_order = osd_order

        self.initialize_decoders()

    def get_probabilities(
        self, code: StabilizerCode, error_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        pi, px, py, pz = self.error_model.probability_distribution(
            code, error_rate
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

    def initialize_decoders(self):
        is_css = self.code.is_css

        if is_css:
            self.z_decoder = bposd_decoder(
                self.code.Hx,
                error_rate=0.05,  # ignore this due to the next parameter
                max_iter=self._max_bp_iter,
                bp_method="msl",
                ms_scaling_factor=0,
                osd_method="osd_cs",  # Choose from: "osd_e", "osd_cs", "osd0"
                osd_order=self._osd_order
            )

            self.x_decoder = bposd_decoder(
                self.code.Hz,
                error_rate=0.05,  # ignore this due to the next parameter
                max_iter=self._max_bp_iter,
                bp_method="msl",
                ms_scaling_factor=0,
                osd_method="osd_cs",  # Choose from: "osd_e", "osd_cs", "osd0"
                osd_order=self._osd_order
            )

        else:
            self.decoder = bposd_decoder(
                self.code.stabilizer_matrix,
                error_rate=0.05,  # ignore this due to the next parameter,
                max_iter=self._max_bp_iter,
                bp_method="msl",
                ms_scaling_factor=0,
                osd_method="osd_cs",  # Choose from: "osd_e", "osd_cs", "osd0"
                osd_order=self._osd_order
            )

    def decode(self, syndrome: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        is_css = self.code.is_css
        n_qubits = self.code.n
        syndrome = np.array(syndrome, dtype=int)

        if is_css:
            syndrome_z = self.code.extract_z_syndrome(syndrome)
            syndrome_x = self.code.extract_x_syndrome(syndrome)

        pi, px, py, pz = self.get_probabilities(self.code, error_rate)

        probabilities_x = px + py
        probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if is_css:
            # Update probabilities (in case the distribution is new at each iteration)
            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)

            # Decode Z errors
            self.z_decoder.decode(syndrome_x)
            z_correction = self.z_decoder.osdw_decoding

            # Bayes update of the probability
            if self._channel_update:
                new_x_probs = self.update_probabilities(
                    z_correction, px, py, pz, direction="z->x"
                )
                self.x_decoder.update_channel_probs(new_x_probs)

            # Decode X errors
            self.x_decoder.decode(syndrome_z)
            x_correction = self.x_decoder.osdw_decoding

            correction = np.concatenate([x_correction, z_correction])
        else:
            # Update probabilities (in case the distribution is new at each iteration)
            self.decoder.update_channel_probs(probabilities)

            # Decode all errors
            self.decoder.decode(syndrome)
            correction = self.decoder.osdw_decoding
            correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])

        return correction


def test_decoder():
    from panqec.codes import XCubeCode
    from panqec.error_models import PauliErrorModel
    import time
    rng = np.random.default_rng()

    L = 10
    code = XCubeCode(L, L, L)

    error_rate = 0.1
    r_x, r_y, r_z = [0.15, 0.15, 0.7]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    decoder = BeliefPropagationOSDDecoder(
        code, error_model, osd_order=10, max_bp_iter=1000
    )

    print("Create stabilizer matrix")
    code.stabilizer_matrix

    print("Create Hx and Hz")
    code.Hx
    code.Hz

    # Start timer
    start = time.time()

    n_iter = 200
    accuracy = 0
    for i in range(n_iter):
        print(f"\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, error_rate, rng=rng)
        print("Calculate syndrome")
        syndrome = code.measure_syndrome(error)
        print("Decode")
        correction = decoder.decode(syndrome, error_rate)
        print("Get total error")
        total_error = (correction + error) % 2

        codespace = code.in_codespace(total_error)
        success = not code.is_logical_error(total_error) and codespace
        print(success)
        accuracy += success

    accuracy /= n_iter
    print("Average time per iteration", (time.time() - start) / n_iter)
    print("Logical error rate", 1 - accuracy)


if __name__ == '__main__':
    test_decoder()
