import panqec.bsparse as bsparse
import numpy as np
from scipy.sparse import csr_matrix


def test_zero_row():
    matrix = bsparse.zero_row(10)

    assert matrix.shape == (1, 10)
    assert matrix.nnz == 0
    assert isinstance(matrix.data, np.ndarray)
    assert isinstance(matrix.indices, np.ndarray)
    assert isinstance(matrix.indptr, np.ndarray)


def test_empty_row():
    matrix = bsparse.empty_row(10)

    assert matrix.shape == (0, 10)
    assert isinstance(matrix.data, np.ndarray)
    assert isinstance(matrix.indices, np.ndarray)
    assert isinstance(matrix.indptr, np.ndarray)


def test_from_array():
    array = np.eye(2)
    spmatrix = bsparse.from_array(array)

    assert spmatrix.shape == (2, 2)
    assert np.all(spmatrix.data == np.array([1, 1]))
    assert np.all(spmatrix.indices == np.array([0, 1]))
    assert np.all(spmatrix.indptr == np.array([0, 1, 2]))


def test_to_array():
    # If we give it an array, it should give back the same array
    array = np.eye(2)
    assert np.all(bsparse.to_array(array) == array)

    # If we give it a spmatrix, it should give the corresponding array
    spmatrix = csr_matrix(array)
    assert np.all(bsparse.to_array(spmatrix) == array)
    assert np.all(bsparse.to_array(spmatrix) == spmatrix.toarray())


def test_is_empty():
    matrix = bsparse.empty_row(10)

    assert bsparse.is_empty(matrix)


def test_is_sparse():
    array = np.array([1, 2, 3])
    matrix = csr_matrix(array)

    assert bsparse.is_sparse(matrix)
    assert not bsparse.is_sparse(array)


def test_is_one():
    matrix = csr_matrix(np.array([1, 0, 1, 1]))

    assert bsparse.is_one(0, matrix)
    assert not bsparse.is_one(1, matrix)
    assert bsparse.is_one(2, matrix)
    assert bsparse.is_one(3, matrix)


def test_insert_mod2():
    matrix = bsparse.zero_row(3)

    bsparse.insert_mod2(2, matrix)

    assert bsparse.is_one(2, matrix)
    assert not bsparse.is_one(0, matrix)
    assert not bsparse.is_one(1, matrix)

    bsparse.insert_mod2(2, matrix)

    assert len(matrix.data) == 0

    bsparse.insert_mod2(0, matrix)
    bsparse.insert_mod2(1, matrix)
    bsparse.insert_mod2(2, matrix)

    assert len(matrix.data) == 3


def test_equal():
    matrix1 = bsparse.from_array([[0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1]])
    matrix1_prime = bsparse.from_array([[0, 1, 0, 1, 0, 0],
                                        [1, 1, 1, 1, 0, 1]])
    matrix1_wrong = bsparse.from_array([[1, 1, 0, 1, 0, 0],
                                        [1, 1, 1, 1, 0, 1]])
    matrix2 = bsparse.from_array(np.ones((5, 10)))
    matrix3 = bsparse.from_array(np.zeros((5, 10)))

    assert bsparse.equal(matrix1, matrix1_prime)
    assert not bsparse.equal(matrix1, matrix1_wrong)
    assert not bsparse.equal(matrix1, matrix2)
    assert not bsparse.equal(matrix1, matrix3)
    assert not bsparse.equal(matrix1, 1)
    assert not bsparse.equal(1, matrix1)
    assert not bsparse.equal(matrix1, 0)
    assert not bsparse.equal(0, matrix1)

    assert bsparse.equal(matrix2, 1)
    assert bsparse.equal(1, matrix2)
    assert not bsparse.equal(matrix2, 0)
    assert not bsparse.equal(0, matrix2)
    assert bsparse.equal(matrix3, 0)
    assert bsparse.equal(0, matrix3)
    assert not bsparse.equal(matrix3, 1)
    assert not bsparse.equal(1, matrix3)


def test_vstack():
    empty_matrix = bsparse.empty_row(2)
    first_row = bsparse.zero_row(2)
    second_row = bsparse.zero_row(2)

    bsparse.insert_mod2(0, first_row)
    bsparse.insert_mod2(1, second_row)

    still_first_row = bsparse.vstack([empty_matrix, first_row])
    new_matrix = bsparse.vstack([first_row, second_row])

    assert bsparse.equal(still_first_row, first_row)
    assert new_matrix.shape == (2, 2)
    assert np.all(bsparse.to_array(new_matrix) == np.eye(2))


def test_hstack():
    full_block = bsparse.from_array([[0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1]])
    a_block = bsparse.from_array([[0, 1, 0], [1, 1, 1]])
    b_block = bsparse.from_array([[1, 0, 0], [1, 0, 1]])

    new_matrix = bsparse.hstack([a_block, b_block])

    assert new_matrix.shape == (a_block.shape[0],
                                a_block.shape[1] + b_block.shape[1])
    assert bsparse.equal(new_matrix, full_block)


def test_hsplit():
    # Test with row vectors
    full_row = bsparse.from_array([1, 0, 0, 1])
    a_row = bsparse.from_array([1, 0])
    b_row = bsparse.from_array([0, 1])

    a, b = bsparse.hsplit(full_row)

    assert bsparse.equal(a_row, a)
    assert bsparse.equal(b_row, b)

    # Test with full matrix
    full_block = bsparse.from_array([[0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1]])
    a_block = bsparse.from_array([[0, 1, 0], [1, 1, 1]])
    b_block = bsparse.from_array([[1, 0, 0], [1, 0, 1]])

    a, b = bsparse.hsplit(full_block)
    a_bis, b_bis = bsparse.hsplit(bsparse.hstack([a_block, b_block]))

    assert bsparse.equal(a_block, a)
    assert bsparse.equal(b_block, b)
    assert bsparse.equal(a_block, a_bis)
    assert bsparse.equal(b_block, b_bis)


def test_dot():
    a = bsparse.zero_row(10)
    b = bsparse.zero_row(10)

    assert bsparse.dot(a, b) == 0
    bsparse.insert_mod2(5, a)
    assert bsparse.dot(a, b) == 0
    bsparse.insert_mod2(5, b)
    assert bsparse.dot(a, b) == 1
    bsparse.insert_mod2(2, a)
    assert bsparse.dot(a, b) == 1
    bsparse.insert_mod2(2, b)
    assert bsparse.dot(a, b) == 0

    # Test that it works with numpy arrays too
    assert bsparse.dot(a.toarray(), b) == 0
    assert bsparse.dot(a.toarray(), b.toarray()) == 0
    assert bsparse.dot(a, b.toarray()) == 0
