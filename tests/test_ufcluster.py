from copy import deepcopy
import pytest
import numpy as np
from panqec.decoders.union_find.clustering import Cluster_Tree, Support, _hash_s_index, _smallest_invalid_cluster

syndrome = np.zeros(5)
syndrome[[1,3]] = 1 #[0, 1, 0, 1, 0]
Hz = np.array([
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
    ]) # random parity matrix
sp_global = Support(syndrome, Hz)

class TestClusterTree:

    def test_get_cluster(self):
        sp = deepcopy(sp_global)
        c = sp.get_cluster(1)
        assert c == Cluster_Tree(1)

    def test_get_clusters(self):
        sp = deepcopy(sp_global)
        cs = sp.get_all_clusters()
        assert set(cs) == {Cluster_Tree(1), Cluster_Tree(3)}
    
    def test_merge(self):
        sp = deepcopy(sp_global)
        c = sp.get_cluster(1)
        c.merge([sp.get_cluster(3)], sp)
        assert c._size == 2
        assert c._odd == False
        assert sp._parents[3] == 1
        assert c._boundary_list == {-2, -4}

    def test_grow(self):
        sp = deepcopy(sp_global)
        c = sp.get_cluster(1)
        assert c.grow(sp) == {2, 3, 7}
        #TODO so many thinking assert c.grow(sp) == {}


class TestSupport:
    def test_support(self):
        sp = deepcopy(sp_global)
        assert sp._num_stabilizer == 5
        assert sp._num_qubit == 10
        assert np.array_equal(sp._H_to_grow, Hz)
        Hz[0][0] = 0
        assert not np.array_equal(sp._H_to_grow, Hz)
        check = {1: Cluster_Tree(1), 3: Cluster_Tree(3)}
        for k, v in sp._cluster_forest.items():
            assert v == check[k]
    
    def test_find_root(self):
        sp = deepcopy(sp_global)
        sp._parents[0] = 1
        sp._parents[1] = 2
        sp._parents[2] = 3
        sp._parents[3] = 3
        assert sp.find_root(2) == 3
        assert sp.find_root(1) == 3
        assert sp._parents[1] == 3
        assert sp._parents[0] == 1
        assert sp.find_root(0) == 3
        assert sp._parents[0] == 3 # compressed
    
    def test_grow_stabilizer(self):
        sp = deepcopy(sp_global)
        nb, fl = sp.grow_stabilizer(0)
        assert nb == [0, 1, 5]
        assert fl == [0, 1, 5]
        assert np.array_equal(sp._H_to_grow[0], np.zeros(10))

    def test_grow_qubit(self):
        sp = deepcopy(sp_global)
        nb, fl = sp.grow_qubit(2)
        assert nb == [1, 2]
        assert fl == [2]
        assert np.array_equal(sp._H_to_grow[:, 2], np.zeros(5))
    
    def test_union(self):
        sp = deepcopy(sp_global)
        return # test merge first
        sp.union([0,1])



def test_hash_index():
    assert _hash_s_index(5) == -6
    assert _hash_s_index(_hash_s_index(5)) == 5

def cust_clstr(rt, odd, size, b_l):
    clst = Cluster_Tree(rt)
    clst._odd = odd
    clst._size = size
    clst._boundary_list = b_l
    return clst

def test_smallest_invalid_cluster():
    # empty clusters 
    clts = {}
    sml, forest = _smallest_invalid_cluster(clts)
    assert sml is None
    assert len(forest) == 0

    # No invalid
    clts = {
        cust_clstr(1, False, 2, []),
        cust_clstr(2, False, 3, []),
        cust_clstr(5, False, 4, [])
    }
    sml, forest = _smallest_invalid_cluster(clts)
    assert sml is None
    assert len(forest) == 0

    # One invalid but size not smallest
    c = cust_clstr(1, True, 4, [])
    clts = {
        cust_clstr(2, False, 1, []),
        c,
        cust_clstr(5, False, 4, [])
    }
    sml, forest = _smallest_invalid_cluster(clts)
    assert sml == c
    assert forest == [c]

    # Multiple Invalid
    c = cust_clstr(1, True, 4, [])
    d = cust_clstr(5, True, 2, [])
    clts = {
        cust_clstr(2, False, 1, []),
        c,
        d
    }
    sml, forest = _smallest_invalid_cluster(clts)
    assert sml == d
    assert set(forest) == set([c,d])   


    

