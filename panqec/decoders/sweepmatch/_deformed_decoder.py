from typing import Tuple, Dict
import numpy as np
from pymatching import Matching
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel, PauliErrorModel
from .. import (
    SweepDecoder3D, Toric3DMatchingDecoder, RotatedPlanarMatchingDecoder,
    RotatedSweepDecoder3D
)


class DeformedToric3DMatchingDecoder(Toric3DMatchingDecoder):
    code: StabilizerCode
    error_rate: float
    error_model: PauliErrorModel
    _epsilon: float
    _n_faces: Dict[str, int]

    def __init__(self, code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)
        self._epsilon = 1e-15
        self._n_faces = dict()

    def new_matcher(self):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).
        n_faces: int = int(3*np.product(self.code.size))
        self._n_faces[self.code.label] = n_faces

        # Only keep the Z vertex stabilizers.
        H_z = self.code.Hz
        weights = self.get_deformed_weights(self.code)
        return Matching(H_z, spacelike_weights=weights)

    def get_deformed_weights(self) -> np.ndarray:
        """Get MWPM weights for deformed Pauli noise."""

        return calculate_deformed_weights(
            self.error_model, self.error_rate, self.code, self._epsilon
        )


class DeformedSweepDecoder3D(SweepDecoder3D):

    code: StabilizerCode
    error_model: BaseErrorModel
    error_rate: float
    _p_edges: int

    def __init__(self, code, error_model, error_rate):
        super().__init__(code, error_model, error_rate)
        self._p_edges = self.get_edge_probabilities()

    def get_edge_probabilities(self):
        """Most likely face for detectable Z error."""

        p_edges: Tuple[float, float, float]

        p_X, p_Y, p_Z = (
            np.array(self.error_model.direction)*self.error_rate
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

    label = 'Deformed Toric 3D Sweep Matching Decoder'
    sweeper: BaseDecoder
    matcher: BaseDecoder

    def __init__(self, code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        self.sweeper = DeformedSweepDecoder3D(
            code, error_model, error_rate
        )
        self._matcher = DeformedToric3DMatchingDecoder(
            code, error_model, error_rate
        )

    def decode(
        self, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        z_correction = self.sweeper.decode(syndrome)
        x_correction = self.matcher.decode(syndrome)

        correction = (z_correction + x_correction) % 2
        correction = correction.astype(np.uint)
        return correction


class DeformedRotatedSweepMatchDecoder(DeformedSweepMatchDecoder):

    def __init__(self, code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        self.sweeper = RotatedSweepDecoder3D(code, error_model, error_rate)
        self.matcher = DeformedRotatedPlanarMatchingDecoder(
            code, error_model, error_rate
        )


class DeformedRotatedPlanarMatchingDecoder(RotatedPlanarMatchingDecoder):

    def __init__(self, code, error_model: BaseErrorModel, error_rate: float):
        self._epsilon = 1e-15
        super().__init__(code, error_model, error_rate)

    def new_matcher(self):
        """Return a new Matching object."""
        # Get the number of X stabilizers (faces).
        n_faces = len([
            location
            for location in self.code.stabilizer_coordinates
            if self.code.stabilizer_type(location) == 'face'
        ])
        self._n_faces[self.code.label] = n_faces
        n_qubits = self.code.n
        self._n_qubits[self.code.label] = n_qubits

        # Only keep the Z vertex stabilizers.
        H_z = self.code.Hz
        weights = self.get_deformed_weights()
        return Matching(H_z, spacelike_weights=weights)

    def get_deformed_weights(self) -> np.ndarray:
        """Get MWPM weights for deformed Pauli noise."""

        return calculate_deformed_weights(
            self.error_model, self.error_rate, self.code, self._epsilon
        )


def calculate_deformed_weights(
    error_model: PauliErrorModel, probability: float,
    code: StabilizerCode, epsilon
) -> np.ndarray:
    regular_weight, deformed_weight = get_regular_and_deformed_weights(
        error_model.direction, probability, epsilon
    )

    # All weights are regular weights to start off with.
    weights = np.ones(code.n, dtype=float)*regular_weight

    # Get indices of deformed qubits from error model.
    deformation_indices = error_model.get_deformation_indices(code)

    # The weights on the deformed edge are different.
    weights[deformation_indices] = deformed_weight

    # Return flattened arrays.
    return weights


def get_regular_and_deformed_weights(
    direction: Tuple[float, float, float], error_rate: float, epsilon: float
) -> Tuple[float, float]:
    """Get MWPM weights for given Pauli noise probabilities."""

    # Extract undeformed error probabilities.
    r_x, r_y, r_z = direction
    p_X, p_Y, p_Z = np.array([r_x, r_y, r_z])*error_rate

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
        (p_regular + epsilon) / (1 - p_regular + epsilon)
    )
    deformed_weight = -np.log(
        (p_deformed + epsilon) / (1 - p_deformed + epsilon)
    )

    return regular_weight, deformed_weight
