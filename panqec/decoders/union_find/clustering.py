from __future__ import annotations
from gettext import find
from operator import xor
from typing import Any, Dict, Set, Tuple, List
from xmlrpc.client import Boolean
import numpy as np
from scipy.sparse import csr_matrix
import sys

def clustering(syndrome: np.ndarray, code_size):
    """Given a syndrome and the code size, returns a correction to apply
    to the qubits

    Parameters
    ----------
    syndrome: np.ndarray
        Syndrome as an array of size m, where m is the number of
        stabilizers. Each element contains 1 if the stabilizer is
        activated and 0 otherwise

    code_size: tuple
        Code size is a tuple that specify the length and the width of the
        toric code.

    Returns
    -------
    correction : np.ndarray
        Correction as an array of size 2n (with n the number of qubits)
        in the binary symplectic format.
    """  
    support = Support(syndrome, code_size)
    
    (smallest_cluster, invalid_clusters) = \
        _smallest_invalid_cluster(support.get_all_clusters())

    while smallest_cluster: # while exists not valid cluster
        fusion_set = smallest_cluster.grow(support) 

        for q in fusion_set:
            Hr = support.H[:, q] - support._H_to_grow[:, q]
            ss = np.where(Hr[:,q] != 0)
            support.union(ss)

            # We don't waste time to check boundary list here, 
            # since it's trivial and it doesn't hurt to have them checked 
            # during runtime

        (smallest_cluster, invalid_clusters) = \
        _smallest_invalid_cluster(support.get_all_clusters())

    return #grwon supprt and syndrome

class Cluster_Tree():
    """Cluster representation"""
    def __init__(self, root):
        self._size = 1
        self._odd = True
        self._root: int = root
        self._boundary_list: set[int] = [root]
    
    def is_invalid(self) -> bool:
        # is_odd is invalid for 2d toric code
        return self._odd

    def is_odd(self) -> bool:
        return self._odd
    
    def get_size(self):
        return self._size
    
    def get_root(self):
        return self._root

    def get_boundary(self):
        return self._boundary_list

    def size_increment(self, inc = 1):
        self._size += inc
    
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
        new_boundary: set = {}
        fusion_set: set = {}
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
            support._parents[rt] = self._root
            self._boundary_list = self._boundary_list.union(c.get_boundary())

def _smallest_invalid_cluster(clts: list[Cluster_Tree]) \
                            -> tuple[Cluster_Tree, list[Cluster_Tree]]:
    """Given a list of cluster tree, 
    reutrns a tuple of the smallest cluster tree and a list of odd cluster trees.

    Parameters
    ----------
    clts: list[Cluster_Tree]
        A list of all cluster trees.

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
            invalids += c
            if c.get_size < minSize:
                sml = c
                minSize = c.get_size
    return (sml, invalids)

def _hash_s_index(i: int):
    return -i-1

class Support():
    """ Storage Class of status and information of the code."""

    def __init__(self, syndrome: np.ndarray, H: csr_matrix):
        self._num_qubit = H.shape[1]
        self._num_stabilizer = H.shape[0]
        self.H = H
        self._H_to_grow = H.copy() # connected only if qubit/stabilizer is not grown 
        self._s_status = syndrome
        self._q_status = np.zeros(self._num_qubit, dtype='uint8') # eraser
        self._parents  = np.full(self._num_qubit, -1, dtype='unit8') # -1 means no parents/it's root
        self._cluster_forest = self._init_cluster_forest(self._syndrome_loc) # stabilizer to cluster
    
    def _init_cluster_forest(self, syndrome: np.ndarray) -> dict[int, Cluster_Tree]:
        """Given an array of syndrome, returns a mapping to its cluster tree.

        Parameters
        ----------
        syndrome: np.array
            ndarray of syndromes

        Returns
        -------
        forest : dict[int, Cluster_Tree]
            The index of a syndrome is mapped to a cluster tree
        """
        forest = {}
        indices = np.where(syndrome != 0)
        for i in indices:
            self._parents[i] = i
            forest[i] = Cluster_Tree(i)
        return forest

    def get_cluster(self, rt: int):
        """Given a root index, returns the Cluster it belongs to"""
        try:
            return self._cluster_forest[rt]
        except KeyError:
            print("The input is not a root vertex!")

    def get_all_clusters(self) -> list:
        """returns all clusters at the current stage."""
        return list(self._cluster_forest.values())

    def find_root(self, v:int) -> int:
        """Given a vertex index, returns the root of the cluster it belongs to."""
        p = self._parents[v]
        seen = [v]
        while p != p:
            v = p
            seen.append(v)
            p = self._parents[v]
        self._parents[seen] = v
        return v

    def union(self, s_l: list[int]):
        """Given a list of indeices of stabilizers, union their cluster.

        Parameters
        ----------
        s_l: 
            List of indicies of stabilizers.

        """
        biggest = None
        clusters = []    
        max = 0
        forest = self._cluster_forest
        for s in s_l:
            rt = self.find_root(s)
            if rt == -1:
                clusters.append(Cluster_Tree(s))
                continue
            c = forest.pop(rt)
            size = c.get_size()
            clusters.append(c)
            if size > max:
                max = size
                biggest = c

        clusters.remove(biggest)
        biggest.merge(clusters, self)
        forest[biggest.get_root()] = biggest

    def grow_stabilizer(self, s: int) -> tuple[np.array, np.array]:
        new_boundary, fusion_list = np.where(self._H_to_grow[s] != 0)
        self._H_to_grow[s] = 0
        return (new_boundary, fusion_list)
        
    def grow_qubit(self, q: int) -> tuple[np.ndarray, np.ndarray]:
        new_boundary = np.where(self._H_to_grow[:, q] != 0)
        self._H_to_grow[:, q] = 0
        return (new_boundary, [q])