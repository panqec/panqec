from __future__ import annotations
from gettext import find
from operator import xor
from typing import Any, Dict, Set, Tuple, List
from xmlrpc.client import Boolean
import numpy as np
from scipy.sparse import csr_matrix
import sys

def clustering(syndrome: np.ndarray, H: csr_matrix):
    """Given a syndrome and the  partial parity check matrix (eg Hz or Hx), returns a list of erasure clusters
    to be fed into the peeling function to obtain a correction for a particular type of error.

    Parameters
    ----------
    syndrome: np.ndarray
        Syndrome as an array of size m, where m is the number of
        stabilizers. Each element contains 1 if the stabilizer is
        activated and 0 otherwise

    H: csr_matrix
        The partial parity check matrix for 1 type of stabilizers
        eg: Hz or Hx.

    Returns
    -------
    output : np.ndarray
        An array of tuples of the form [([],[]), ([],[]) ..]. Each 
        tuple holds the information of one of the clusters generated.
        The tuples are such that the first element is a list of
        stabilizers (indices) and the second element is a list of 
        qubits (indices) for that particular cluster.

    """  
    support = Support(syndrome, H)
    
    (smallest_cluster, invalid_clusters) = \
        _smallest_invalid_cluster(support.get_all_clusters())

    while smallest_cluster: # while exists not valid cluster
        fusion_set = smallest_cluster.grow(support) 

        for q in fusion_set:
            Hr = support.H[:, q] - support._H_to_grow[:, q]
            ss = list(Hr.nonzero()[0])
            support._q_parents[q] = support.union(ss)

            # We don't waste time to check boundary list here, 
            # since it's trivial and it doesn't hurt to have them checked 
            # during runtime

        (smallest_cluster, invalid_clusters) = \
        _smallest_invalid_cluster(support.get_all_clusters())

    return support.to_peeling()

class Cluster_Tree():
    """Cluster representation"""
    def __init__(self, root, odd=True):
        self._size = 1
        self._odd = odd
        self._root: int = root
        self._boundary_list: set[int] = set([_hash_s_index(root)])
    
    def is_invalid(self) -> bool:
        # is_odd is invalid for 2d toric code
        return self._odd

    def is_odd(self) -> bool:
        return self._odd
    
    def get_size(self):
        return self._size
    
    def get_root(self):
        return self._root

    def get_boundary(self) -> set[int]:
        return self._boundary_list
    
    def merge(self, clusters: list[Cluster_Tree], support: Support):
        """Given a cluster tree, merge it with the current instance of cluster.

        Parameters
        ----------
        clusters: list[Cluster_Tree]
            A list of cluster tree representation
        """ 
        for c in clusters:
            self._size += c.get_size()
            self._odd = xor(self._odd, c.is_odd())
            rt = c.get_root()
            support._s_parents[rt] = self._root
            self._boundary_list = self._boundary_list.union(c.get_boundary())    
    
    def grow(self, support: Support) -> set[int]:
        """Given a support, grow every vertex in the boundary list,
        returns the fusion set of qubits.

        Parameters
        ----------
        support: Support
            The support class instance of the code.

        Returns
        -------
        fusion set : set[int]
            fusion set as a set of indices of qubits
        """ 
        new_boundary: set = set()
        fusion_set: set = set()
        for b in self._boundary_list:
            if b < 0: # Stabilizer
                nb, fs = support.grow_stabilizer(_hash_s_index(b))               
            else: # Qubit
                nb, fs =support.grow_qubit(b)
                nb = [_hash_s_index(b) for b in nb]
            new_boundary = new_boundary.union(nb)
            fusion_set = fusion_set.union(fs)
        
        self._boundary_list = new_boundary # check if correct after fusion
        return fusion_set

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Cluster_Tree):
            return False
        return __o.get_root() == self._root
    
    def __hash__(self) -> int:
        return hash(self._root)

def _smallest_invalid_cluster(clts: set[Cluster_Tree]) \
                            -> tuple[Cluster_Tree, list[Cluster_Tree]]:
    """Given a list of cluster tree, 
    reutrns a tuple of the smallest cluster tree and a list of odd cluster trees.

    Parameters
    ----------
    clts: set[Cluster_Tree]
        A set of all cluster trees.

    Returns
    -------
    smallest_cluster_tree : Cluster_Tree
        The smallest invalid cluster tree
    
    invalid_trees : list[Cluster_Tree]
        A list of all invalid clutser trees.

    """ 
    sml = None
    invalids = []
    minSize = sys.maxsize
    for c in clts:
        if c.is_invalid():
            invalids.append(c)
            if c.get_size() < minSize:
                sml = c
                minSize = c.get_size()
    return (sml, invalids)

def _hash_s_index(i: int):
    return -i-1

class Support():
    """ Storage Class of status and information of the code."""

    def __init__(self, syndrome: np.ndarray, H: csr_matrix):
        self._num_stabilizer = H.shape[0]
        self._num_qubit = H.shape[1]
        self.H = H
        self._H_to_grow = H.copy() # connected only if qubit/stabilizer is not grown 
        # UNUSE self._s_status = syndrome
        # UNUSE self._q_status = np.zeros(self._num_qubit, dtype='uint8') # eraser
        self._q_parents = np.full(self._num_qubit, -1) # -1 means no parents/it's root
        self._s_parents = np.full(self._num_stabilizer, -1) # -1 means no parents/it's root
        self._cluster_forest = self._init_cluster_forest(syndrome) # stabilizer to cluster
    
    def _init_cluster_forest(self, syndrome: np.ndarray) -> dict[int, Cluster_Tree]:
        """Given an array of syndrome, returns a mapping to its cluster tree.

        Parameters
        ----------
        syndrome: np.ndarray
            ndarray of syndromes

        Returns
        -------
        forest : dict[int, Cluster_Tree]
            The index of a syndrome is mapped to a cluster tree
        """
        forest = {}
        indices = list(np.where(syndrome != 0)[0])
        for i in indices:
            self._s_parents[i] = i
            forest[i] = Cluster_Tree(i)
        return forest

    def get_cluster(self, rt: int):
        """Given a root index, returns the Cluster it belongs to"""
        try:
            return self._cluster_forest[rt]
        except KeyError:
            print("The input is not a root vertex!")

    def get_all_clusters_roots(self) -> list[int]:
        return self._cluster_forest.keys()

    def get_all_clusters(self) -> list:
        """returns all clusters at the current stage."""
        return list(self._cluster_forest.values())

    def find_root(self, v:int) -> int:
        """Given a vertex index, returns the root of the cluster it belongs to."""
        parents = self._s_parents
        p = parents[v]
        if p == -1:
            return -1
        seen = []
        while v != p:
            seen.append(v)
            v = p
            p = parents[v]
        parents[seen] = v
        return v

    def grow_stabilizer(self, s: int) -> tuple[list[int], list[int]]:
        new_boundary = fusion_list = list(self._H_to_grow.getrow(s).nonzero()[1]) # 1 due to get column index
        self._H_to_grow[s] = 0
        return (new_boundary, fusion_list)
        
    def grow_qubit(self, q: int) -> tuple[list[int], list[int]]:
        new_boundary = list(self._H_to_grow.getcol(q).nonzero()[0]) # 0 due to get row index
        self._H_to_grow[:, q] = 0
        return (new_boundary, [q])
    
    def union(self, s_l: list[int]) -> int:
        """Given a list of indeices of stabilizers, union their cluster.

        Parameters
        ----------
        s_l: 
            List of indicies of stabilizers.
        
        Returns
        --------
        root:
            Index of the root of the cluster

        """
        biggest = None
        clusters = []    
        max = 0
        forest = self._cluster_forest
        keys = self.get_all_clusters_roots()
        for s in s_l:
            rt = self.find_root(s)
            if rt == -1:
                clusters.append(Cluster_Tree(s, odd=False))
                continue
            elif rt not in keys: # have popped
                continue
            c = forest.pop(rt)
            size = c.get_size()
            clusters.append(c)
            if size > max:
                max = size
                biggest = c

        clusters.remove(biggest)
        biggest.merge(clusters, self)
        root = biggest.get_root()
        forest[root] = biggest
        return root

    def _update_parents(self, parents: list[int]):
        l = len(parents)
        rs = self.get_all_clusters_roots()
        for i in range(l):
            p = parents[i]
            if p not in rs and p != -1:
                parents[i] = self.find_root(p)

    def to_peeling(self) -> list[tuple[np.ndarray, np.ndarray]]:
        roots = self.get_all_clusters_roots()
        s_parents = self._s_parents
        q_parents = self._q_parents
        self._update_parents(s_parents)
        self._update_parents(q_parents)
        ret = []
        for r in roots:
            # indices not [0,1] array atm
            s = np.where(s_parents == r)[0]
            q = np.where(q_parents == r)[0] 
            ret.append((s, q))
        return ret
