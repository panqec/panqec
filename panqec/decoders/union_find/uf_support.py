from __future__ import annotations

import copy
import sys
from operator import xor
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


class Clustering_Tree():
    """Cluster representation"""

    def __init__(self, root, support: Support, odd=True):
        self._size = 1
        self._odd = odd
        self._root: int = root
        self.support = support
        self._boundary_list: set[int] = set([self.support._hash_s_index(root)])

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

    def merge(self, clusters: list[Clustering_Tree]):
        """Given a list of cluster tree, merge it with the current instance of
        cluster.

        Parameters
        ----------
        clusters: list[Clustering_Tree]
            A list of cluster tree representation
        """
        for c in clusters:
            self._size += c.get_size()
            self._odd = xor(self._odd, c.is_odd())
            rt = c.get_root()
            self.support._s_parents[rt] = self._root
            self._boundary_list = self._boundary_list.union(c.get_boundary())

    def grow(self) -> set[int]:
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
            if b < 0:  # Stabilizer
                nb, fs = self.support.grow_stabilizer(
                    self.support._hash_s_index(b))
            else:  # Qubit
                nb, fs = self.support.grow_qubit(b)
                nb = [self.support. _hash_s_index(b) for b in nb]
            new_boundary = new_boundary.union(nb)
            fusion_set = fusion_set.union(fs)

        self._boundary_list = new_boundary  # check if correct after fusion
        return fusion_set

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Clustering_Tree):
            return False
        return __o.get_root() == self._root

    def __hash__(self) -> int:
        return hash(self._root)


class Peeling_Tree():
    """The tree representation of the peeling step."""

    def __init__(
        self, root: int,
        support: Support,
        stabilizers_ind: np.ndarray,
        qubits_ind: np.ndarray
    ):
        self._root = root
        self._support = support
        self.peeled = False
        self.stablizers = np.zeros(support._num_stabilizer, dtype=bool)
        self.stablizers[stabilizers_ind] = 1
        self.qubits = np.zeros(support._num_qubit, dtype=bool)
        self.qubits[qubits_ind] = 1
        self.H = csr_matrix(support.H.toarray() & self.qubits.reshape(
            # subgraph for connection between stabilizers and qubits
            1, -1) & self.stablizers.reshape(-1, 1))

        self.syndrome = support._s_status.copy().astype(bool)
        self.syndrome[~self.stablizers.astype(bool)] = 0
        self._stablizers_connections, self._leaves = self._build_tree(
            self.H, self.stablizers, root)

    @staticmethod
    def _build_tree(
        H: csr_matrix, stabilizers: np.ndarray, root: int
    ) -> tuple[csr_matrix, list[int]]:
        """Given the parity matrix, stabilizers and qubits, builds and returns
            the tree representation using BSF spanning tree.

        Parameters
        ----------
        H (readonly): csr_matrix
            The parity matrix of the code.

        stabilizers: np.array[bool]
            List of indices of stabilizers.

        root: int
            The index of the root of the tree.

        Returns
        -------

        S: csr_matrix
            Asymmetric matrix as the tree representation among stabilizers.

        leaves: list[int]
            The array of the indices of the stabilizer as the leaves.
        """

        # stabilizer connection in the sub graph through qubits
        S = (H @ H.T).astype(bool)
        leaves_ind = [root]
        # remove copy if we don't need stabilizer anymore
        unseen = copy.deepcopy(stabilizers)
        unseen[root] = 0

        while np.sum(unseen) > 0:
            new_leaves_ind = []
            for s in leaves_ind:
                children = S[s].toarray().reshape(-1).astype(bool)
                if not np.any(children):
                    new_leaves_ind.append(s)
                    continue

                # remove other potential 'parents' to its childrem
                S[:, children] = 0
                S[children, s] = 0  # remove the child-to-parent connection
                S[s, children] = 1  # add back the parent-to-child connection

                new_leaves_ind.extend(np.where(children & unseen)[0])
                unseen[children] = 0

            leaves_ind = new_leaves_ind
        S.setdiag(0)
        return S, leaves_ind

    def _update_syndrome(
            self,
            parents: np.ndarray,
            curr_leaves_ind: list[int],
            syndrome: np.ndarray):
        """
        Given the parents and leaves, returns the mapping of
        parents to children.

        Parameters
        ----------
        parents: np.ndarray
            List of indices of parents.

        curr_leaves_ind: np.ndarray
            List of bool indicating if a leaf is a syndrome.

        syndrome: np.ndarray
            List of the syndrome.

        Returns
        -------
        Update the syndrome in-place.
        """
        syndrome_leaves = syndrome[curr_leaves_ind]
        parent_to_syndrome: dict[int, int] = {}
        for p, sl in zip(parents, syndrome_leaves):
            new_syn = parent_to_syndrome.get(p, syndrome[p])
            parent_to_syndrome[p] = new_syn != sl

        for p, syn in parent_to_syndrome.items():
            syndrome[p] = syn

        syndrome[curr_leaves_ind] = False

    def peel(self) -> list[int]:
        """Peel the tree and return the correction.

        Returns
        -------
        correction: list[int]
            The correction as a list of indices of qubits."""
        if self.peeled:
            return []

        correction = []
        child_to_p = self._stablizers_connections.T
        curr_syndromes = self.syndrome
        curr_leaves_ind = self._leaves
        while np.sum(curr_syndromes) > 0:
            parents = np.where(child_to_p[curr_leaves_ind].toarray())[
                1]  # indice of parrents for current leaves
            parent_qubits = self.H[parents, :].toarray().astype(bool)
            leaf_qubits = self.H[curr_leaves_ind, :].toarray().astype(bool)
            syndrome_leaves = curr_syndromes[curr_leaves_ind]
            correction.extend(np.where((parent_qubits & leaf_qubits)[
                              syndrome_leaves, :])[1].tolist())
            self._update_syndrome(parents, curr_leaves_ind, curr_syndromes)
            child_to_p[curr_leaves_ind, :] = 0
            curr_leaves_ind = np.unique(np.array(parents)[np.where(
                (~child_to_p.toarray())[:, parents].all(axis=0))[0]])
        self.peeled = True
        return correction


class Support():
    """ Storage Class of status and information of the code,
    for matrix operations."""

    def __init__(self, syndrome: np.ndarray, H: csr_matrix):
        """
        Parameters
        ----------
        syndrome: np.ndarray
            Syndrome as an array of size m, where m is the number of
            stabilizers. Each element contains 1 if the stabilizer is
            activated and 0 otherwise

        H: csr_matrix
            The partial parity check matrix for 1 type of stabilizers
            eg: Hz or Hx.
        """
        self._num_stabilizer = H.shape[0]
        self._num_qubit = H.shape[1]
        self.H = H  # save original conncection
        # connected only if qubit/stabilizer is not grown
        self._H_to_grow = H.copy()
        self._s_status = syndrome
        # eraser
        # UNUSE self._q_status = np.zeros(self._num_qubit, dtype='uint8')
        # -1 means no parents/it's root
        self._q_parents = np.full(self._num_qubit, -1)
        # -1 means no parents/it's root
        self._s_parents = np.full(self._num_stabilizer, -1)

    def _init_cluster_forest(self) -> dict[int, Clustering_Tree]:
        """Given an array of syndrome, returns a mapping to its cluster tree.

        Parameters
        ----------
        syndrome: np.ndarray
            ndarray of syndromes

        Returns
        -------
        forest : dict[int, Clustering_Tree]
            The index of a syndrome is mapped to a cluster tree
        """
        forest = {}
        indices = list(np.where(self._s_status != 0)[0])
        for i in indices:
            self._s_parents[i] = i
            forest[i] = Clustering_Tree(i, self)
        return forest

    def find_root(self, v: int) -> int:
        """
            Given a vertex index,
            returns the root of the cluster it belongs to.
        """
        parents = self._s_parents
        p = parents[v]
        if p == -1:
            return -1
        seen = []
        while v != p:
            seen.append(v)
            v = p
            p = parents[v]
        # update parent of seen nodes to be root.
        parents[seen] = v
        return v

    def grow_stabilizer(self, s: int) -> tuple[list[int], list[int]]:
        new_boundary = fusion_list = list(self._H_to_grow.getrow(s).nonzero()[
                                          1])  # 1 due to get column index
        self._H_to_grow[s] = 0
        return (new_boundary, fusion_list)

    def grow_qubit(self, q: int) -> tuple[list[int], list[int]]:
        new_boundary = list(self._H_to_grow.getcol(
            q).nonzero()[0])  # 0 due to get row index
        self._H_to_grow[:, q] = 0
        return (new_boundary, [q])

    def merge_clusters(
        self,
        s_l: list,
        cluster_forest: dict[int, Clustering_Tree]
    ) -> int:
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
        keys = cluster_forest.keys()
        for s in s_l:
            rt = self.find_root(s)
            if rt == -1:
                # dummy cluster for the new stabilizer just grown
                clusters.append(Clustering_Tree(s, self, odd=False))
                continue
            elif rt not in keys:  # have popped
                continue
            c = cluster_forest.pop(rt)
            size = c.get_size()
            clusters.append(c)
            if size > max:
                max = size
                biggest = c

        if biggest is None:
            return -1

        clusters.remove(biggest)
        biggest.merge(clusters)
        root = biggest.get_root()
        cluster_forest[root] = biggest
        return root

    def clustering(self):
        """
        Given the syndrome and parity matrix, returns the clusters generated.

        Returns
        -------
        None
        output : np.ndarray
            An array of tuples of the form [([],[]), ([],[]) ..]. Each
            tuple holds the information of one of the clusters generated.
            The tuples are such that the first element is a list of
            stabilizers (indices) and the second element is a list of
            qubits (indices) for that particular cluster.

        """
        cluster_forest = self._init_cluster_forest()

        (smallest_cluster, invalid_clusters) = \
            self._smallest_invalid_cluster(set(cluster_forest.values()))

        while smallest_cluster:  # while exists not valid cluster
            fusion_set = smallest_cluster.grow()
            # print(f"smlest cluster: {smallest_cluster}")
            # print(fusion_set)

            for q in fusion_set:
                Hr = self.H[:, q] - self._H_to_grow[:, q]
                ss = list(Hr.nonzero()[0])
                self._q_parents[q] = self.merge_clusters(ss, cluster_forest)

                # We don't waste time to check boundary list here,
                # since it's trivial and it doesn't hurt to have them checked
                # during runtime

            (smallest_cluster, invalid_clusters) = \
                self._smallest_invalid_cluster(set(cluster_forest.values()))

        roots = set(list(cluster_forest.keys()))
        self._update_parents(self._s_parents, roots)
        self._update_parents(self._q_parents, roots)

        return roots

    def _update_parents(self, parents: np.ndarray, roots: set[int]):
        for i in range(len(parents)):
            p = parents[i]
            if p not in roots and p != -1:
                parents[i] = self.find_root(p)

    @staticmethod
    def _smallest_invalid_cluster(clts: set[Clustering_Tree]) \
            -> tuple[Optional[Clustering_Tree], list[Clustering_Tree]]:
        """
            Given a list of cluster tree,
            reutrns a tuple of the smallest cluster tree
            and a list of odd cluster trees.

        Parameters
        ----------
        clts: set[Clustering_Tree]
            A set of all cluster trees.

        Returns
        -------
        smallest_cluster_tree : Clustering_Tree
            The smallest invalid cluster tree

        invalid_trees : list[Clustering_Tree]
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

    @staticmethod
    def _hash_s_index(i: int):
        return -i-1

    def peeling(self, roots: set[int]):
        correction_ind = []
        for r in roots:
            s = np.where(self._s_parents == r)[0]
            q = np.where(self._q_parents == r)[0]
            pt = Peeling_Tree(r, self, s, q)
            correction_ind.extend(pt.peel())
        correction = np.zeros(self._num_qubit, dtype='uint8')
        correction[correction_ind] = 1
        return correction

    def decode(self):
        roots = self.clustering()
        return self.peeling(roots)
