"""
Noise models for 3D Toric Code including deformed noise.

:Author:
    Eric Huang
"""

import functools
import itertools
from typing import Tuple
import numpy as np
from qecsim import paulitools as pt
from qecsim.models.generic import SimpleErrorModel
from qecsim.model import ErrorModel, StabilizerCode
from .bpauli import barray_to_bvector, bvector_to_barray, get_bvector_index
from .utils import nested_map


class PauliErrorModel(SimpleErrorModel):
    """Pauli channel IID noise model."""

    # direction: Tuple[float, float, float]

    def __init__(self, r_x, r_y, r_z):
        if not np.isclose(r_x + r_y + r_z, 1):
            raise ValueError(
                f'Noise direction ({r_x}, {r_y}, {r_z}) does not sum to 1.0'
            )
        self._direction = r_x, r_y, r_z

    @property
    def direction(self):
        return self._direction

    @property
    def label(self):
        return 'Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(*self.direction)

    def generate(self, code: StabilizerCode, probability: float, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n
        p_i, p_x, p_y, p_z = self.probability_distribution(code, probability)

        error_pauli = ''.join([rng.choice(
            ('I', 'X', 'Y', 'Z'),
            p=[p_i[i], p_x[i], p_y[i], p_z[i]]
        ) for i in range(n_qubits)])

        bsf = pt.pauli_to_bsf(error_pauli)

        return bsf

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, probability: float
    ) -> Tuple:
        n = code.n
        r_x, r_y, r_z = self.direction

        p_i = (1 - probability) * np.ones(n)
        p_x = (r_x * probability) * np.ones(n)
        p_y = (r_y * probability) * np.ones(n)
        p_z = (r_z * probability) * np.ones(n)

        return p_i, p_x, p_y, p_z


class XNoiseOnYZEdgesOnly(ErrorModel):
    """IID Bit flip noise on y and z edges only.

    No errors on x edges.
    """

    label: str = 'X on yz edges'
    direction: Tuple[float, float, float]

    def __init__(self):
        self.direction = (1, 0, 0)

    def generate(
        self, code: StabilizerCode, probability: float, rng=None
    ) -> np.ndarray:
        """Sample noise."""

        # Initialize a Pauli operator object.
        pauli_op = dict()

        # 0 means no error, 1 means error.
        choices = [0, 1]

        # Probabilities of no error and probabilit of error.
        probabilities = [1 - probability, probability]

        # Generate errors on y edge and z edge.
        y_edge_errors = rng.choice(choices, size=code.size, p=probabilities)
        z_edge_errors = rng.choice(choices, size=code.size, p=probabilities)

        for x, y, z in code.qubit_index.keys():
            if y_edge_errors[x, y, z]:
                pauli_op[(x, y, z)] = 'X'
            if z_edge_errors[x, y, z]:
                pauli_op[(x, y, z)] = 'X'

        # Convert to binary sympectic form.
        error = code.to_bsf(pauli_op)
        return error


def generate_pauli_noise(
    p_X: float, p_Y: float, p_Z: float, L: int
) -> np.ndarray:
    operators = np.random.choice(
        np.arange(4, dtype=int),
        size=(3, L, L, L),
        p=(1 - p_X - p_Y - p_Z, p_X, p_Y, p_Z)
    ).tolist()

    def int_to_two_bit(x):
        return {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}[x]

    operators = np.array(
        nested_map(int_to_two_bit)(operators),
        dtype=np.uint
    )
    bvector = barray_to_bvector(operators, L)

    return bvector


def deform_operator(bvector: np.ndarray, L: int, edge: int = 0) -> np.ndarray:
    """Conjugate with Hadamard all operators acting on edge."""
    operator = bvector_to_barray(bvector, L)
    deformed_op = operator.copy()

    # Swap the X block and Z block parts for sites to be deformed.
    for x in range(L):
        for y in range(L):
            for z in range(L):
                deformed_op[edge, x, y, z, 0] = operator[edge, x, y, z, 1]
                deformed_op[edge, x, y, z, 1] = operator[edge, x, y, z, 0]

    deformed_bvector = barray_to_bvector(deformed_op, L)
    return deformed_bvector


def get_deformed_weights(
    p_X: float, p_Y: float, p_Z: float, L: int, epsilon: float = 1e-15
) -> np.ndarray:
    """Get MWPM weights for deformed Pauli noise."""

    # The x-edges are deformed.
    deformed_edge = 0
    p_regular = p_X + p_Y
    p_deformed = p_Z + p_Y

    regular_weight = -np.log(p_regular + epsilon/(1 - p_regular + epsilon))
    deformed_weight = -np.log(p_deformed + epsilon/(1 - p_deformed + epsilon))

    # All weights are regular weights to start off.
    weights = np.ones(3*L**3, dtype=float)*regular_weight

    # Modify the weights on the special edge.
    for x in range(L):
        for y in range(L):
            for z in range(L):
                index = get_bvector_index(deformed_edge, x, y, z, 0, L)
                weights[index] = deformed_weight

    return weights


def get_direction_from_bias_ratio(pauli: str, eta) -> dict:
    """Get noise params given Pauli and bias."""

    if eta == np.inf:
        r_bias = 1.
    else:
        r_bias = eta / (1 + eta)
    r_other = (1 - r_bias) / 2

    params: dict = {}

    if pauli == 'Z':
        params = {'r_x': r_other, 'r_y': r_other, 'r_z': r_bias}
    elif pauli == 'X':
        params = {'r_x': r_bias, 'r_y': r_other, 'r_z': r_other}
    elif pauli == 'Y':
        params = {'r_x': r_other, 'r_y': r_bias, 'r_z': r_other}

    return params
