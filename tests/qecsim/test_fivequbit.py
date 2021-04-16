"""
Test five qubit code very simple run.
"""

import pytest
import numpy as np
from qecsim.models.basic import FiveQubitCode
from qecsim.paulitools import pauli_to_bsf


@pytest.fixture
def five_qubit_code():
    code = FiveQubitCode()
    return code


def test_stabilizers(five_qubit_code):
    code = five_qubit_code
    expected_stabilizers = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
    assert np.all(code.stabilizers == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_stabilizers
    ]))


def test_logicals(five_qubit_code):
    code = five_qubit_code
    expected_logical_xs = ['XXXXX']
    expected_logical_zs = ['ZZZZZ']
    assert np.all(code.logical_xs == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_logical_xs
    ]))
    assert np.all(code.logical_zs == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_logical_zs
    ]))
