from typing import Tuple
import itertools
import numpy as np
from pymatching import Matching
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from .. import (
    SweepDecoder3D, Toric3DPymatchingDecoder, RotatedPlanarPymatchingDecoder,
    RotatedSweepDecoder3D
)
import panqec.bsparse as bsparse


class DeformedToric3DPymatchingDecoder(Toric3DPymatchingDecoder):

    _error_model: BaseErrorModel
    _probability: float
    _epsilon: float

    def __init__(self, error_model: BaseErrorModel, probability: float):
        super(DeformedToric3DPymatchingDecoder, self).__init__()
        self._error_model = error_model
        self._probability = probability
        self._epsilon = 1e-15

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).
        n_faces = int(np.product(code.shape))
        self._n_faces[code.label] = n_faces
        n_qubits = code.n

        # Only keep the Z vertex stabilizers.
        H_z = code.stabilizer_matrix[n_faces:, n_qubits:]
        weights = self.get_deformed_weights(code)
        return Matching(H_z, spacelike_weights=weights)

    def get_deformed_weights(self, code: StabilizerCode) -> np.ndarray:
        """Get MWPM weights for deformed Pauli noise."""

        # The x-edges are deformed.
        deformed_edge = code.X_AXIS

        regular_weight, deformed_weight = get_regular_and_deformed_weights(
            self._error_model.direction, self._probability, self._epsilon
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


class DeformedSweepDecoder3D(SweepDecoder3D):

    _error_model: BaseErrorModel
    _probability: float
    _p_edges: int

    def __init__(self, error_model, probability):
        super(DeformedSweepDecoder3D, self).__init__()
        self._error_model = error_model
        self._probability = probability
        self._p_edges = self.get_edge_probabilities()

    def get_edge_probabilities(self):
        """Most likely face for detectable Z error."""

        p_edges: Tuple[float, float, float]

        p_X, p_Y, p_Z = (
            np.array(self._error_model.direction)*self._probability
        )
        p_regular = p_Y + p_Z
        p_deformed = p_Y + p_X

        p = np.array([p_deformed, p_regular, p_regular])
        p_edges = tuple(p/p.sum())

        return p_edges

    def get_default_direction(self):
        """Use most likely direction based on noise."""
        direction = int(self._rng.choice([0, 1, 2], size=1, p=self._p_edges))
        return direction


class DeformedSweepMatchDecoder(BaseDecoder):

    label = 'Deformed Toric 3D Sweep Pymatching Decoder'
    _sweeper: DeformedSweepDecoder3D
    _matcher: DeformedToric3DPymatchingDecoder

    def __init__(self, error_model: BaseErrorModel, probability: float):
        self._sweeper = DeformedSweepDecoder3D(
            error_model, probability
        )
        self._matcher = DeformedToric3DPymatchingDecoder(
            error_model, probability
        )

    def decode(self, code: StabilizerCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self._sweeper.decode(code, syndrome)
        x_correction = self._matcher.decode(code, syndrome)

        correction = (z_correction + x_correction) % 2
        correction = correction.astype(np.uint)
        return bsparse.from_array(correction)


class DeformedRotatedSweepMatchDecoder(DeformedSweepMatchDecoder):

    def __init__(self, error_model: BaseErrorModel, probability: float):
        self._sweeper = RotatedSweepDecoder3D()
        self._matcher = DeformedRotatedPlanarPymatchingDecoder(
            error_model, probability
        )


class DeformedRotatedPlanarPymatchingDecoder(RotatedPlanarPymatchingDecoder):

    def __init__(self, error_model: BaseErrorModel, probability: float):
        super(DeformedRotatedPlanarPymatchingDecoder, self).__init__()
        self._error_model = error_model
        self._probability = probability
        self._epsilon = 1e-15

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).
        n_faces = len(code.face_index)
        self._n_faces[code.label] = n_faces
        n_qubits = code.n
        self._n_qubits[code.label] = n_qubits

        # Only keep the Z vertex stabilizers.
        H_z = code.stabilizer_matrix[n_faces:, n_qubits:]
        weights = self.get_deformed_weights(code)
        return Matching(H_z, spacelike_weights=weights)

    def get_deformed_weights(self, code: StabilizerCode) -> np.ndarray:
        """Get MWPM weights for deformed Pauli noise."""

        regular_weight, deformed_weight = get_regular_and_deformed_weights(
            self._error_model.direction, self._probability, self._epsilon
        )

        # Count qubits and faces.
        n_qubits = code.n

        # All weights are regular weights to start off with.
        weights = np.ones(n_qubits, dtype=float)*regular_weight

        # The weights on the deformed edge are different.
        for i_qubit, (x, y, z) in code.qubit_coordinates:
            if z % 2 == 0:
                weights[i_qubit] = deformed_weight

        # Return flattened arrays.
        return weights


def get_regular_and_deformed_weights(
    direction: Tuple[float, float, float], probability: float, epsilon: float
) -> Tuple[float, float]:
    """Get MWPM weights for given Pauli noise probabilities."""

    # Extract undeformed error probabilities.
    r_x, r_y, r_z = direction
    p_X, p_Y, p_Z = np.array([r_x, r_y, r_z])*probability

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

    return regular_weight, deformed_weight
