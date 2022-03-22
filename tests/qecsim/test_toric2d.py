"""
Test 2D toric code using qecsim.

:Author:
    Eric Huang
"""

import pytest
import numpy as np
from bn3d.models import Toric2DCode, Toric2DPauli
from qecsim.models.toric import ToricMWPMDecoder
from qecsim.paulitools import bsp


@pytest.fixture
def lattice_size():
    L_x = 4
    L_y = 5
    n_qubits = 2*L_x*L_y
    return L_x, L_y, n_qubits


@pytest.fixture
def code(lattice_size):
    L_x, L_y, _ = lattice_size
    return Toric2DCode(L_x, L_y)


@pytest.fixture
def decoder():
    return ToricMWPMDecoder()


def test_general_properties(code, lattice_size):
    L_x, L_y, n_qubits = lattice_size
    assert code.label == 'Toric 4x5'
    assert code.shape == (2, L_x, L_y)
    assert code.n_k_d == (n_qubits, 2, min(L_x, L_y))
    assert len(code.stabilizers) == 2*L_x*L_y
    assert len(code.logical_xs) == 2
    assert len(code.logical_zs) == 2


def test_stabilizers_commute(code, lattice_size):
    L_x, L_y, n_qubits = lattice_size
    assert code.stabilizers.shape == (n_qubits, 2*n_qubits)
    commutations = np.array([
        bsp(stabilizer_1, stabilizer_2)
        for stabilizer_1 in code.stabilizers
        for stabilizer_2 in code.stabilizers
    ])
    assert np.all(commutations == 0)


def test_stabilizers_commute_with_logicals(code):
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


def test_logicals_anticommute_correctly(code):
    commutations = np.array([
        [
            bsp(logical_x, logical_z)
            for logical_z in code.logical_zs
        ]
        for logical_x in code.logical_xs
    ])
    assert np.all(commutations == np.identity(2))


def test_ascii_art(code):
    operator = Toric2DPauli(code, code.logical_xs[0])
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


def test_syndrome_calculation(code):

    # Construct a single-qubit X error somewhere.
    error = Toric2DPauli(code)
    error.site('X', (1, 2, 3))
    assert code.ascii_art(pauli=error) == (
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   X   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  '
    )

    # Calculate the syndrome.
    syndrome = bsp(error.to_bsf(), code.stabilizers.T)
    assert code.ascii_art(syndrome=syndrome) == (
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │   │  \n'
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │   │  \n'
        '┼───┼───┼───┼───┼──\n'
        '│   │   │ Z │ Z │  \n'
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │   │  '
    )


def test_mwpm_decoder(code, decoder):
    error = Toric2DPauli(code)
    error.site('X', (1, 2, 3))
    error.site('X', (1, 2, 2))
    error.site('X', (0, 2, 1))
    error.site('X', (0, 1, 1))
    error.site('X', (1, 0, 2))
    error.site('X', (1, 0, 3))
    error.site('X', (0, 1, 3))
    assert code.ascii_art(pauli=error) == (
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   X   X   ·  \n'
        '┼─·─┼─X─┼─·─┼─X─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─X─┼─·─┼─·─┼─·\n'
        '·   ·   X   X   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  '
    )

    syndrome = bsp(error.to_bsf(), code.stabilizers.T)
    assert code.ascii_art(syndrome=syndrome) == (
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │   │  \n'
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │ Z │  \n'
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │ Z │  \n'
        '┼───┼───┼───┼───┼──\n'
        '│   │   │   │   │  '
    )

    correction = decoder.decode(code, syndrome)
    assert code.ascii_art(pauli=code.new_pauli(bsf=correction)) == (
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─·─┼─X─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  '
    )

    total_error = (error.to_bsf() + correction) % 2
    assert code.ascii_art(pauli=code.new_pauli(bsf=total_error)) == (
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   X   X   ·  \n'
        '┼─·─┼─X─┼─·─┼─X─┼─·\n'
        '·   ·   ·   ·   ·  \n'
        '┼─·─┼─X─┼─·─┼─X─┼─·\n'
        '·   ·   X   X   ·  \n'
        '┼─·─┼─·─┼─·─┼─·─┼─·\n'
        '·   ·   ·   ·   ·  '
    )

    # Test the total error commutes with the stabilizer so we are back
    # in the code space.
    assert all([
        bsp(total_error, stabilizer) == 0
        for stabilizer in code.stabilizers
    ])

    # Test that no logical error has occured either.
    assert all([
        bsp(total_error, logical_x) == 0
        for logical_x in code.logical_xs
    ])
    assert all([
        bsp(total_error, logical_z) == 0
        for logical_z in code.logical_zs
    ])
