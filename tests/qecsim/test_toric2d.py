"""
Test 2D toric code.
"""

import pytest
import numpy as np
from qecsim.models.toric import ToricCode, ToricPauli
from qecsim.paulitools import bsf_to_pauli, bsp


@pytest.fixture
def lattice_size():
    L_x = 4
    L_y = 5
    n_qubits = 2*L_x*L_y
    return L_x, L_y, n_qubits


@pytest.fixture
def toric_code(lattice_size):
    L_x, L_y, _ = lattice_size
    code = ToricCode(L_x, L_y)
    return code


def test_general_properties(toric_code, lattice_size):
    code = toric_code
    L_x, L_y, n_qubits = lattice_size
    assert code.label == 'Toric 4x5'
    assert code.shape == (2, L_x, L_y)
    assert code.n_k_d == (n_qubits, 2, min(L_x, L_y))
    assert len(code.stabilizers) == 2*L_x*L_y
    assert len(code.logical_xs) == 2
    assert len(code.logical_zs) == 2


def test_stabilizers_commute(toric_code, lattice_size):
    code = toric_code
    L_x, L_y, n_qubits = lattice_size
    assert code.stabilizers.shape == (n_qubits, 2*n_qubits)
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

    logical_z_stabilizer_commutations = np.array([
        bsp(stabilizer, logical_z)
        for stabilizer in code.stabilizers
        for logical_z in code.logical_zs
    ])
    assert np.all(logical_z_stabilizer_commutations == 0)


def test_logicals_anticommute_correctly(toric_code):
    code = toric_code
    commutations = np.array([
        [
            bsp(logical_x, logical_z)
            for logical_z in code.logical_zs
        ]
        for logical_x in code.logical_xs
    ])
    assert np.all(commutations == np.identity(2))


def test_ascii_art(toric_code):
    code = toric_code
    operator = ToricPauli(code, code.logical_xs[0])
    print(code.ascii_art(pauli=operator))
    print(repr(code.ascii_art(pauli=operator)))
    assert code.ascii_art(pauli=operator) == (
        '┼─·─┼─·─┼─X─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─X─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─X─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─X─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  '
    )
