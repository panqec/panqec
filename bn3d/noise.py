"""
Noise models for 3D Toric Code.
"""

import numpy as np
from .bpauli import barray_to_bvector, bvector_to_barray
from .utils import nested_map


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
