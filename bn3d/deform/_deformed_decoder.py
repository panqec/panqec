import itertools
import numpy as np
from pymatching import Matching
from qecsim.model import Decoder, StabilizerCode, ErrorModel
from ..tc3d import SweepDecoder3D, Toric3DPymatchingDecoder


class DeformedToric3DPymatchingDecoder(Toric3DPymatchingDecoder):

    _error_model: ErrorModel
    _probability: float
    _epsilon: float

    def __init__(self, error_model: ErrorModel, probability: float):
        super(DeformedToric3DPymatchingDecoder, self).__init__()
        self._error_model = error_model
        self._probability = probability
        self._epsilon = 1e-15

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).
        n_faces = int(np.product(code.shape))
        self._n_faces[code.label] = n_faces
        n_qubits = code.n_k_d[0]

        # Only keep the Z vertex stabilizers.
        H_z = code.stabilizers[n_faces:, n_qubits:]
        weights = self.get_deformed_weights(code)
        print(f'{set(weights)=}')
        return Matching(H_z, spacelike_weights=weights)

    def get_deformed_weights(self, code: StabilizerCode) -> np.ndarray:
        """Get MWPM weights for deformed Pauli noise."""

        # Extract undeformed error probabilities.
        r_x, r_y, r_z = self._error_model.direction
        p_X, p_Y, p_Z = np.array([r_x, r_y, r_z])*self._probability
        print(f'{r_x=} {r_y=} {r_z=} {self._probability=}')

        # Very small regularizer to deal with infinities.
        epsilon = self._epsilon

        # The x-edges are deformed.
        deformed_edge = code.X_AXIS

        # For undeformed qubit sites, only X and Y errors can be detected,
        # so the probability of error is the sum of their probabilities.
        # Note that Z errors can neither be detected nor corrected so they
        # do not contribute to the weight.
        p_regular = p_X + p_Y

        # For deformed qubit sites, only Z and Y errors can be detected.
        p_deformed = p_Z + p_Y

        # Take logarithms regularized by epsilons to avoid infinities.
        # Logarithms turn products into sums.
        # Divide by the probability of no (detectable) error because that is
        # the baseline to compare with.
        regular_weight = -np.log(
            (p_regular + epsilon)
            / (1 - p_regular + epsilon)
        )
        deformed_weight = -np.log(
            (p_deformed + epsilon)
            / (1 - p_deformed + epsilon)
        )

        # All weights are regular weights to start off with.
        weights = np.ones(code.shape, dtype=float)*regular_weight

        # The ranges of indices to iterate over.
        ranges = [range(length) for length in code.shape]

        # The weights on the deformed edge are different.
        for axis, x, y, z in itertools.product(*ranges):
            if axis == deformed_edge:
                weights[axis, x, y, z] = deformed_weight

        # Return flattened arrays.
        return weights.flatten()


class DeformedSweepMatchDecoder(Decoder):

    label = 'Toric 3D Sweep Pymatching Decoder'
    _sweeper: SweepDecoder3D
    _matcher: DeformedToric3DPymatchingDecoder

    def __init__(self, error_model: ErrorModel, probability: float):
        self._sweeper = SweepDecoder3D()
        self._matcher = DeformedToric3DPymatchingDecoder(
            error_model, probability
        )

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self._sweeper.decode(code, syndrome)
        x_correction = self._matcher.decode(code, syndrome)

        correction = (z_correction + x_correction) % 2
        return correction
