"""
Tests to make sure qecsim pauli bit strings are working as expected.

:Author:
    Eric Huang
"""
import numpy as np
import qecsim
from qecsim.paulitools import pauli_to_bsf, bsf_to_pauli, bsp


def test_qecsim_version():
    assert qecsim.__version__ is not None


def test_pauli_to_bsf():
    assert np.all(
        pauli_to_bsf('IXYZ') == [0, 1, 1, 0, 0, 0, 1, 1]
    )


def test_bvector_to_pauli_string():
    assert (
        bsf_to_pauli(np.array([0, 1, 1, 0, 0, 0, 1, 1])) == 'IXYZ'
    )


def test_pauli_string_bvector_inverse():
    pstring = 'IXYZ'
    bvector = pauli_to_bsf(pstring)
    new_pstring = bsf_to_pauli(bvector)
    assert pstring == new_pstring


def test_bsp():
    III = np.array([0, 0, 0, 0, 0, 0])
    XXX = np.array([1, 1, 1, 0, 0, 0])
    ZZZ = np.array([0, 0, 0, 1, 1, 1])
    assert bsp(XXX, ZZZ) == 1
    assert bsp(III, XXX) == 0
    assert bsp(III, XXX) == 0
