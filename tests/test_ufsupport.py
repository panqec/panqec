from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix

from panqec.decoders.union_find.uf_support import Clustering_Tree, Support

syndrome = np.zeros(5)
syndrome[[1, 3]] = 1  # [0, 1, 0, 1, 0]
Hz = csr_matrix([
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
])  # random parity matrix
sp_global = Support(syndrome, Hz)


class TestClusterTree:

    def test_merge(self):
        sp = deepcopy(sp_global)
        c1 = Clustering_Tree(1, sp)
        c3 = Clustering_Tree(3, sp)
        c1.merge([c3])
        assert c1._size == 2
        assert not c1._odd
        assert sp._s_parents[3] == 1
        assert c1._boundary_list == {-2, -4}

    def test_grow(self):
        sp = deepcopy(sp_global)
        c = Clustering_Tree(1, sp)
        assert c.grow() == {2, 3, 7}
        assert c.get_boundary() == {2, 3, 7}
        assert c.grow() == {2, 3, 7}
        assert c.get_boundary() == {-3, -4}
        assert c.grow() == {4, 5, 6}
        assert c.get_boundary() == {4, 5, 6}
        assert c.grow() == {4, 5, 6}
        assert c.get_boundary() == {-1, -5}
        assert c.grow() == {0, 1, 8, 9}
        assert c.get_boundary() == {0, 1, 8, 9}
        assert c.grow() == {0, 1, 8, 9}
        assert c.get_boundary() == set()
        assert c.grow() == set()
        assert c.get_boundary() == set()


class TestSupport:
    def test_support(self):
        sp = deepcopy(sp_global)
        assert sp._num_stabilizer == 5
        assert sp._num_qubit == 10
        assert np.array_equal(sp._H_to_grow.toarray(), Hz.toarray())
        Hz[0][0] = 0
        assert not np.array_equal(sp._H_to_grow, Hz.toarray())
        # check = {1: Clustering_Tree(1, sp), 3: Clustering_Tree(3, sp)}
        # for k, v in sp._cluster_forest.items():
        #     assert v == check[k]

    def test_find_root(self):
        sp = deepcopy(sp_global)
        sp._s_parents[0] = 1
        sp._s_parents[1] = 2
        sp._s_parents[2] = 3
        sp._s_parents[3] = 3
        assert sp.find_root(2) == 3
        assert sp.find_root(1) == 3
        assert sp._s_parents[1] == 3
        assert sp._s_parents[0] == 1
        assert sp.find_root(0) == 3
        assert sp._s_parents[0] == 3  # compressed

    def test_grow_stabilizer(self):
        sp = deepcopy(sp_global)
        nb, fl = sp.grow_stabilizer(0)
        assert nb == [0, 1, 5]
        assert fl == [0, 1, 5]
        assert np.array_equal(sp._H_to_grow[0].toarray()[0], np.zeros(10))

    def test_grow_qubit(self):
        sp = deepcopy(sp_global)
        nb, fl = sp.grow_qubit(2)
        assert nb == [1, 2]
        assert fl == [2]
        assert np.array_equal(sp._H_to_grow.toarray()[:, 2], np.zeros(5))

    def test_update_parents(self):
        sp = deepcopy(sp_global)
        sp._s_parents[1] = 1
        sp._s_parents[2] = 1
        sp._s_parents[3] = 3
        sp._s_parents[4] = 2
        sp._update_parents(sp._s_parents, set([1, 3]))
        assert list(sp._s_parents) == [-1, 1, 1, 3, 1]
        assert np.array_equal(sp._q_parents, np.full(10, -1))
        sp._q_parents[1] = 3
        sp._q_parents[3] = 3
        sp._q_parents[4] = 2
        sp._update_parents(sp._q_parents, set([1, 3]))
        assert list(sp._q_parents) == [-1, 3, -1, 3, 1, -1, -1, -1, -1, -1]

    def test_clustering(self):
        sp = deepcopy(sp_global)
        init_forest = sp._init_cluster_forest()
        assert len(init_forest) == 2
        tp = sp.clustering()
        assert len(tp) == 1
        assert tp == set([1])
