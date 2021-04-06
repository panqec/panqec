import numpy as np
from bn3d.bpauli import bcommute
from bn3d.tc3d import (
    get_vertex_Z_stabilisers, get_face_X_stabilisers, get_all_stabilisers,
    get_Z_logicals, get_X_logicals, get_all_logicals,
)


def test_get_vertex_Z_stabilisers():
    L = 3
    stabilisers = get_vertex_Z_stabilisers(L)
    assert stabilisers.dtype == np.uint

    # Weight of every stabiliser should be 6.
    assert np.all(stabilisers.sum(axis=1) == 6)

    # Number of stabiliser generators should be number of vertices.
    assert stabilisers.shape[0] == L**3

    # The number of qubits should be the number of edges 3L^3.
    assert stabilisers.shape[1] == 2*3*L**3

    # There should be no X or Y operators.
    assert np.all(stabilisers[:, :3*L**3] == 0)

    # Each qubit should be in the support of exactly 2 stabilisers.
    assert np.all(stabilisers.sum(axis=0)[3*L**3:] == 2)


def test_get_face_X_stabilisers():
    L = 3
    stabilisers = get_face_X_stabilisers(L)

    # Weight of every stabiliser should be 6.
    assert np.all(stabilisers.sum(axis=1) == 4)
    assert stabilisers.dtype == np.uint

    # Number of stabiliser generators should be number of edges.
    assert stabilisers.shape[0] == 3*L**3

    # The number of qubits should be the number of edges 3L^3.
    assert stabilisers.shape[1] == 2*3*L**3

    # There should be no Z or Y operators.
    assert np.all(stabilisers[:, 3*L**3:] == 0)

    # Each qubit should be in the support of exactly 4 stabilisers.
    assert np.all(stabilisers.sum(axis=0)[:3*L**3] == 4)


def test_get_all_stabilisers():
    L = 3
    stabilisers = get_all_stabilisers(3)

    # Total number of stabilisers.
    assert stabilisers.shape[0] == 4*3*L**2
    # Z block of X stabilisers should be all 0.
    assert np.all(stabilisers[:3*L**3, 3*L**3:] == 0)

    # X block of Z stabilisers should be all 0.
    assert np.all(stabilisers[3*L**3:, :L**3] == 0)


def test_get_Z_logicals():
    L = 3
    logicals = get_Z_logicals(L)
    assert logicals.shape[0] == 3
    assert logicals.shape[1] == 2*3*L**3


def test_get_X_logicals():
    L = 3
    logicals = get_X_logicals(L)
    assert logicals.shape[0] == 3
    assert logicals.shape[1] == 2*3*L**3


class TestCommutationRelations:

    def test_stabilisers_commute_with_each_other(self):
        L = 3
        stabilisers = get_all_stabilisers(L)
        assert np.all(bcommute(stabilisers, stabilisers) == 0)

    def test_Z_logicals_commute_with_each_other(self):
        L = 3
        logicals = get_Z_logicals(L)
        assert np.all(bcommute(logicals, logicals) == 0)

    def test_X_logicals_commute_with_each_other(self):
        L = 3
        logicals = get_X_logicals(L)
        assert np.all(bcommute(logicals, logicals) == 0)

    def test_stabilisers_commute_with_logicals(self):
        L = 3
        stabilisers = get_all_stabilisers(L)
        logicals = get_all_logicals(L)
        assert np.all(bcommute(logicals, stabilisers) == 0)

    def test_X_and_Z_logicals_commutation(self):
        L = 3
        X_logicals = get_X_logicals(L)
        Z_logicals = get_Z_logicals(L)
        commutation = bcommute(X_logicals, Z_logicals)
        assert np.all(commutation == np.identity(L))
