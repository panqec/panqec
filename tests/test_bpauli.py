import numpy as np
from bn3d.bpauli import (
    pauli_string_to_bvector, bvector_to_pauli_string, bvector_to_barray,
    barray_to_bvector, bcommute
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


def test_bcommute_singles():
    III = np.array([0, 0, 0, 0, 0, 0])
    XXX = np.array([1, 1, 1, 0, 0, 0])
    ZZZ = np.array([0, 0, 0, 1, 1, 1])
    assert bcommute(XXX, ZZZ) == 1
    assert bcommute(III, XXX) == 0
    assert bcommute(III, XXX) == 0


def test_bcommute_one_to_many():
    XYZ = pauli_string_to_bvector('XYZ')
    IXY = pauli_string_to_bvector('IXY')
    ZZI = pauli_string_to_bvector('ZZI')
    assert bcommute(XYZ, IXY) == 0
    assert bcommute(IXY, ZZI) == 1
    assert np.all(bcommute(XYZ, [IXY, ZZI]) == [[0, 0]])
    assert np.all(bcommute([XYZ, IXY], [ZZI, IXY]) == [[0, 0], [1, 0]])


def test_bvector_to_barray():
    L = 3
    np.random.seed(0)
    a = np.random.rand(3, L, L, L, 2)
    assert a.shape == (3, L, L, L, 2)
    s = barray_to_bvector(a, L)
    assert s.shape[0] == 2*3*L**3
    a_new = bvector_to_barray(s, L)
    assert np.all(a == a_new)
