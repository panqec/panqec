"""
Tests for custom implementation of pauli bit strings.

:Author:
    Eric Huang
"""
import pytest
import numpy as np
from bn3d.bpauli import (
    pauli_string_to_bvector, bvector_to_pauli_string, bvector_to_barray,
    barray_to_bvector, bcommute, get_effective_error, bvector_to_int,
    bvectors_to_ints, ints_to_bvectors
)


def test_pauli_string_to_bvector():
    assert np.all(
        pauli_string_to_bvector('IXYZ') == [0, 1, 1, 0, 0, 0, 1, 1]
    )


def test_bvector_to_pauli_string():
    assert (
        bvector_to_pauli_string(np.array([0, 1, 1, 0, 0, 0, 1, 1])) == 'IXYZ'
    )


def test_pauli_string_bvector_inverse():
    pstring = 'IXYZ'
    bvector = pauli_string_to_bvector(pstring)
    new_pstring = bvector_to_pauli_string(bvector)
    assert pstring == new_pstring


class TestBcommute:

    def test_bcommute_singles(self):
        III = np.array([0, 0, 0, 0, 0, 0])
        XXX = np.array([1, 1, 1, 0, 0, 0])
        ZZZ = np.array([0, 0, 0, 1, 1, 1])
        assert bcommute(XXX, ZZZ) == 1
        assert bcommute(III, XXX) == 0
        assert bcommute(III, XXX) == 0

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


def test_get_effective_errror_single():
    logicals = np.array([
        pauli_string_to_bvector('XXXXX'),
        pauli_string_to_bvector('ZZZZZ')
    ])
    total_error = pauli_string_to_bvector('YYYYY')
    effective_error = get_effective_error(logicals, total_error)
    assert np.all(effective_error.shape == (2, ))
    assert bvector_to_pauli_string(effective_error) == 'Y'


def test_get_effective_errror_many():
    logicals = np.array([
        pauli_string_to_bvector('XXXXX'),
        pauli_string_to_bvector('ZZZZZ')
    ])
    total_error = np.array([
        pauli_string_to_bvector('YYYYY'),
        pauli_string_to_bvector('IIIII'),
        pauli_string_to_bvector('XXXZZ'),
    ])
    effective_error = get_effective_error(logicals, total_error)
    assert np.all(effective_error.shape == (3, 2))
    assert np.all(effective_error == np.array([
        pauli_string_to_bvector('Y'),
        pauli_string_to_bvector('I'),
        pauli_string_to_bvector('X')
    ]))


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
