import pytest
import numpy as np
from panqec.decoders.union_find.clustering import Cluster_Tree, Support, _hash_s_index, _smallest_invalid_cluster

class TestClusterTree:
    pass

class TestSupport:

    def test_support(self):
        syndrome = np.zeros(5)
        syndrome[[1,3]] = 1
        Hz = np.zeros([5,10])
        sp = Support(syndrome, Hz)
        assert sp._num_stabilizer == 5
        assert sp._num_qubit == 10
        assert np.array_equal(sp._H_to_grow, Hz)
        check = {1: Cluster_Tree(1), 3: Cluster_Tree(3)}
        for k, v in sp._cluster_forest.items():
            assert v == check[k]
            

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


    

