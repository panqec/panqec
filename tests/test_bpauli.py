"""
Tests for custom implementation of pauli bit strings.

:Author:
    Eric Huang
"""
import pytest
import numpy as np
from qecsim.models.toric import ToricCode, ToricPauli
from bn3d.bpauli import (
    pauli_string_to_bvector, bvector_to_pauli_string, bvector_to_barray,
    barray_to_bvector, bcommute, get_effective_error, bvector_to_int,
    bvectors_to_ints, ints_to_bvectors
)


def test_pauli_string_to_bvector():
    assert np.all(
        pauli_string_to_bvector('IXYZ') == [0, 1, 1, 0, 0, 0, 1, 1]
    ), 'bsf of IXYZ should be 01100011'


def test_bvector_to_pauli_string():
    assert (
        bvector_to_pauli_string(np.array([0, 1, 1, 0, 0, 0, 1, 1])) == 'IXYZ'
    ), 'Pauli string of 01100011 should be IXYZ'


def test_pauli_string_bvector_inverse():
    pstring = 'IXYZ'
    bvector = pauli_string_to_bvector(pstring)
    new_pstring = bvector_to_pauli_string(bvector)
    assert pstring == new_pstring, (
        'Converting Pauli string to bsf and back should '
        'give original Pauli string'
    )


class TestBcommute:

    def test_bcommute_singles(self):
        III = np.array([0, 0, 0, 0, 0, 0])
        XXX = np.array([1, 1, 1, 0, 0, 0])
        ZZZ = np.array([0, 0, 0, 1, 1, 1])
        assert bcommute(XXX, ZZZ) == 1, 'XXX should anticommute with ZZZ'
        assert bcommute(III, XXX) == 0, 'III should commute with XXX'
        assert bcommute(III, XXX) == 0, 'III should commute with XXX'

    def test_bcommute_many_to_one(self):
        stabilizers = np.array([
            pauli_string_to_bvector('XXI'),
            pauli_string_to_bvector('IXX'),
        ])
        error = pauli_string_to_bvector('IZI')
        syndrome = bcommute(stabilizers, error)
        assert syndrome.shape == (2,), 'syndrome should have shape 2'
        assert np.all(syndrome == [1, 1]), 'IZI should anticommute with both'

        # Changing the order shouldn't matter.
        syndrome = bcommute(error, stabilizers)
        assert syndrome.shape == (2,)
        assert np.all(syndrome == [1, 1])

    def test_bcommute_one_to_many(self):
        XYZ = pauli_string_to_bvector('XYZ')
        IXY = pauli_string_to_bvector('IXY')
        ZZI = pauli_string_to_bvector('ZZI')
        assert bcommute(XYZ, IXY) == 0
        assert bcommute(IXY, ZZI) == 1
        assert np.all(bcommute(XYZ, [IXY, ZZI]) == [[0, 0]])
        assert np.all(bcommute([XYZ, IXY], [ZZI, IXY]) == [[0, 0], [1, 0]])

    def test_raise_error_if_not_even_length(self):
        with pytest.raises(ValueError):
            bcommute([0, 0, 1, 0, 1], [0, 1, 0, 1, 0])

        with pytest.raises(ValueError):
            bcommute([0, 0, 0], [0, 1])

        with pytest.raises(ValueError):
            bcommute([0, 0, 0, 0], [0, 1, 0])

    def test_raise_error_if_unequal_shapes(self):
        with pytest.raises(ValueError):
            bcommute([0, 0, 0, 1], [1, 0, 1, 1, 0, 1])


def test_bvector_to_barray():
    L = 3
    np.random.seed(0)
    a = np.random.rand(3, L, L, L, 2)
    assert a.shape == (3, L, L, L, 2)
    s = barray_to_bvector(a, L)
    assert s.shape[0] == 2*3*L**3
    a_new = bvector_to_barray(s, L)
    assert np.all(a == a_new)


def test_effective_error_wrong_size_raises_exception():
    X_logicals = np.array([1, 0, 1], dtype=np.uint)
    Z_logicals = np.array([1, 0, 1, 0], dtype=np.uint)
    total_error = np.array([0, 1], dtype=np.uint)
    with pytest.raises(ValueError):
        get_effective_error(total_error, X_logicals, Z_logicals)


def test_get_effective_errror_single():
    X_logicals = pauli_string_to_bvector('XXXXX')
    Z_logicals = pauli_string_to_bvector('ZZZZZ')
    total_error = pauli_string_to_bvector('YYYYY')
    effective_error = get_effective_error(total_error, X_logicals, Z_logicals)
    assert np.all(effective_error.shape == (2, ))
    assert bvector_to_pauli_string(effective_error) == 'Y'


def test_get_effective_error_many():
    X_logicals = pauli_string_to_bvector('XXXXX')
    Z_logicals = pauli_string_to_bvector('ZZZZZ')
    total_error = np.array([
        pauli_string_to_bvector('YYYYY'),
        pauli_string_to_bvector('IIIII'),
        pauli_string_to_bvector('XXXZZ'),
    ])
    effective_error = get_effective_error(total_error, X_logicals, Z_logicals)
    assert np.all(effective_error.shape == (3, 2)), (
        'Effective errors should have shape (3, 2), '
        f'instead got {effective_error.shape}'
    )
    assert np.all(effective_error == np.array([
        pauli_string_to_bvector('Y'),
        pauli_string_to_bvector('I'),
        pauli_string_to_bvector('X')
    ])), 'Effective errors should be bsf for Y, I, Z'


def test_bvector_to_int():
    assert bvector_to_int(pauli_string_to_bvector('IIIII')) == 0
    assert bvector_to_int(pauli_string_to_bvector('I')) == 0
    assert bvector_to_int(pauli_string_to_bvector('X')) == 2
    assert bvector_to_int(pauli_string_to_bvector('Y')) == 3
    assert bvector_to_int(pauli_string_to_bvector('Z')) == 1


def test_bvectors_to_ints():
    assert bvectors_to_ints(list(map(
        pauli_string_to_bvector,
        ['III', 'XYZ', 'IIZ']
    ))) == [0, 51, 1]


def test_ints_to_bvectors():
    assert np.all(
        np.array(ints_to_bvectors([0, 1, 2], 3))
        == np.array(list(map(
            pauli_string_to_bvector, ['III', 'IIZ', 'IZI']
        )))
    )


def test_get_effective_error_toric_code_logicals():
    code = ToricCode(3, 5)
    logical_operators = {
        'X1': code.logical_xs[0],
        'X2': code.logical_xs[1],
        'Z1': code.logical_zs[0],
        'Z2': code.logical_zs[1],
    }
    expected_effective_errors = {
        'X1': [1, 0, 0, 0],
        'X2': [0, 1, 0, 0],
        'Z1': [0, 0, 1, 0],
        'Z2': [0, 0, 0, 1],
    }
    for logical in ['X1', 'X2', 'Z1', 'Z2']:
        effective_error = get_effective_error(
            logical_operators[logical], code.logical_xs, code.logical_zs
        )
        assert np.all(effective_error == expected_effective_errors[logical]), (
            f'Logical operator {logical} should have effective error '
            f'{expected_effective_errors[logical]}, '
            f'instead got {effective_error.tolist()}'
        )


def test_get_effective_error_toric_code_logicals_many():
    code = ToricCode(3, 5)
    logical_operators = np.array([
        code.logical_xs[0],
        code.logical_xs[1],
        code.logical_zs[0],
        code.logical_zs[1],
    ])
    expected_effective_errors = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    effective_errors = get_effective_error(
        logical_operators, code.logical_xs, code.logical_zs
    )
    assert np.all(effective_errors == expected_effective_errors), (
        f'Logical operators {logical_operators} should have effective errors '
        f'{expected_effective_errors}, '
        f'instead got {effective_errors}'
    )
