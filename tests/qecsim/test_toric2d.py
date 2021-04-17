"""
Test 2D toric code.
"""

import pytest
import numpy as np
from qecsim.models.toric import ToricCode
from qecsim.paulitools import bsf_to_pauli, bsp


@pytest.fixture
def toric_code():
    L_x = 4
    L_y = 5
    code = ToricCode(L_x, L_y)
    return code


def test_general_properties(toric_code):
    code = toric_code
    assert code.label == 'Toric 4x5'
    assert code.shape == (2, 4, 5)
    assert code.n_k_d == (40, 2, 4)


def test_stabilizers_commute(toric_code):
    code = toric_code
    assert code.stabilizers.shape == (40, 80)
    commutations = np.array([
        bsp(stabilizer_1, stabilizer_2)
        for stabilizer_1 in code.stabilizers
        for stabilizer_2 in code.stabilizers
    ])
    assert np.all(commutations == 0)


def test_stabilizers_commute_with_logicals(toric_code):
    code = toric_code
    logical_x_stabilizer_commutations = np.array([
        bsp(stabilizer, logical_x)
        for stabilizer in code.stabilizers
        for logical_x in code.logical_xs
    ])
    assert np.all(logical_x_stabilizer_commutations == 0)
