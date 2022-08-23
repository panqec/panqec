from __future__ import annotations
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

    (smallest_cluster, odd_clusters) = \
        _smallest_odd_cluster(support.get_all_clusters())

    while smallest_cluster: 
        old_root = smallest_cluster.get_root()
        fusion_list = smallest_cluster.grow() 
        
        # TODO improve: only grow one edge at a time
        # result: no fusion list, only fusion edge
        for e in fusion_list:
            if e.is_fusion_edge():
                support.union(e.fst, e.snd)

        root = old_root.find_root() # the root of cluster after union
        support.get_cluster(root).update_boundary(support)

        (smallest_cluster, odd_clusters) = \
        _smallest_odd_cluster(odd_clusters)

    return #grwon supprt and syndrome

class Cluster_Tree():
    """Cluster representation"""
    def __init__(self, root):
        self._size = 1
        self._odd = True
        self._root: int = root
        self._boundary_list: Set[int] = [root]
    
    def is_odd(self) -> Boolean:
        return self._odd
    
    def get_size(self):
        return self._size
    
    def get_root(self):
        return self._root

    def get_boundary(self):
        return self._boundary_list

    def size_increment(self, inc = 1):
        self._size += inc
    
    def grow(self, support: Support):
        """Given a support, grow every vertex in the boundary list,
        returns the fusion list of edges.

        Parameters
        ----------
        support: Support
            The support class instance of the code.

        Returns
        -------
        correction : np.ndarray
            Correction as an array of size 2n (with n the number of qubits)
            in the binary symplectic format.
        """ 
        new_boundary = []
        fusion_list = []
        for v in self._boundary_list:
            (fl, nb) = support.grow_vertex(v)
            new_boundary += nb
            fusion_list += fl
        
        self._boundary_list.union(new_boundary) # append the boundary vertex set
        return fusion_list
    
    def merge(self, clst: Cluster_Tree):
        """Given a cluster tree, merge it with the current instance of cluster.

        Parameters
        ----------
        clst: Cluster_Tree
            A cluster tree representation
        """ 
        self._size += clst.get_size()
        self._odd = xor(self._odd, clst.is_odd())
        rt = clst.get_root()
        rt.set_parent(self._root)
        #self._root.add_child(rt)
        self._boundary_list.union(clst.get_boundary())
        
    def update_boundary(self, support: Support):
        """ Eliminate the non-boundary vertx in the list"""
        self._boundary_list = set(filter(lambda b : support.vertex_is_boundary(b),
                                     self._boundary_list))

class Vertex():
    """ Vertex representation for stabilizer"""
    def __init__(self,
                 location: Tuple,
                 parent: Vertex = None,
                 children: List[Vertex] = []):
        self._location = location
        self._parent = parent
        self._children = set(children)
    
    def get_location(self):
        """ returns coordinates of the vertex location"""
        return self._location
    
    def add_child(self, child):
        """
            Add new children, return root for cluster size increase.
        """
        self._children.add(child)
        return self.find_root()
    
    def remove_child(self, child):
        self._children.remove(child)

    def set_parent(self, new_parent):
        self._parent = new_parent
    
    def get_parent(self):
        return self._parent

    def find_root(self) -> Vertex:
        """ returns the root vertex of the current vertex,
        compress the seen vertices along the way.
        """
        v = self
        p = v.get_parent()
        seen_v = []
        while p is not None:
            seen_v.append(v)
            v = p
            p = v.get_parent()
        self._compress(v, seen_v)
        return v
        
    def _compress(r: Vertex, l: List[Vertex]):
        for v in l:
            # r.add_child(v)
            v.set_parent(r)

class Edge():
    """Edge representaions for physical quibits"""
    def __init__(self, location):
        self.fst: Vertex = None # The edge grows from this vertex
        self.snd: Vertex = None 
        self._location = location
    
    def get_location(self) -> Tuple:
        """ Returns the coordinates of the edge."""
        return self._location

    def add_vertex(self, fst: Vertex = None, snd: Vertex = None):
        """ Add the attachted vertices sequentially. """ 
        self.fst = fst
        self.snd = snd
    
    def is_fusion_edge(self) -> Boolean:
        """ return if the edge fuese two different clusters"""
        try:
            return self.fst.find_root() is not self.snd.find_root()
        except AttributeError:
            print("The edge is not attached to 2 vertices!")

def _smallest_odd_cluster(clts: List[Cluster_Tree]) \
                            -> Tuple[Cluster_Tree, List[Cluster_Tree]]:
    """Given a list of cluster tree, 
    reutrns a tuple of the smallest cluster tree and a list of odd cluster trees.

    Parameters
    ----------
    clts: List[Cluster_Tree]
        A list of all cluster trees.

    Returns
    -------
    smallest_cluster_tree : Cluster_Tree
        The smallest odd cluster tree
    
    odd_trees : List[Cluster_Tree]
        A list of all odd clutser trees.

    """ 
    sml = None
    odds = []
    minSize = sys.maxsize
    for c in clts:
        if c.is_odd():
            odds += c
            if c.get_size < minSize:
                sml = c
                minSize = c.get_size
    return (sml, odds)


class Support():
    """ Storage Class of status and information of the code."""
    # edges(qubits) status
    UNOCCUPIED = 0
    HALF_GROWN = 1
    GROWN      = 2

    # vertex(stabilizer) status
    DARK_POINT = 0
    VERTEX     = 1
    SYNDROME   = 2

    def __init__(self, syndrome: np.ndarray, H: csr_matrix):
        self._num_qubit = H.shape[1]
        self._num_stabilizer = H.shape[0]
        self._s_status = syndrome
        self._q_status = np.zeros(self._num_qubit, dtype='uint8') # eraser
        self._parents  = np.full(self._num_qubit, -1, dtype='unit8') # -1 means no parents/it's root
        # self._loc_edge_map = {}
        # self._syndrome_loc = []
        # self._loc_vertex_map = self._init_loc_syndrome(syndrome)
        self._loc_cluster_map = self._init_loc_cluster(self._syndrome_loc)
    
    def _init_loc_cluster(self, syndrome: np.ndarray) -> Dict[int, Cluster_Tree]:
        """Given an array of syndrome, returns a mapping to its cluster tree.

        Parameters
        ----------
        syndrome: np.array
            ndarray of syndromes

        Returns
        -------
        forest : Dict[int, Cluster_Tree]
            The index of a syndrome is mapped to a cluster tree
        """
        forest = {}
        for i, s in enumerate(syndrome):
            if s:
                forest[i] = Cluster_Tree(i)
        return forest

    
    def get_cluster(self, rt: int):
        """Given a root index, returns the Cluster it belongs to"""
        try:
            return self._loc_cluster_map[rt]
        except KeyError:
            print("The input is not a root vertex!")

    def get_all_clusters(self) -> list:
        """returns all clusters at the current stage."""
        return list(self._loc_cluster_map.values())

    def vertex_is_boundary(self, s: int):
        """Given a stabilizer index, returns true if it is a boundary vertex"""
        # A vertex is boundary if all edges surrounded are grown.
        return [] == \
            filter(lambda s : s != Support.GROWN, \
                self._status[self._get_surrond_edges(s)])

    def find_root(self, v:int) -> int:
        """Given a vertex index, returns the root of the cluster it belongs to."""
        p = self._parents[v]
        while p != -1:
            v = p
            p = self._parents[v]
        return v

    def union(self, v: int, u: int):
        """Given two root vertices, union their cluster.

        Parameters
        ----------
        rt_v, rt_u: Vertex
            Vertex representation of the root of different clusters.

        """
        rt_v = self.find_root(v)
        rt_u = self.find_root(u)

        clst_v = self._loc_cluster_map[rt_v]
        clst_u = self._loc_cluster_map[rt_u]

        big, sml = clst_u, clst_v if clst_u.get_size() > clst_v.get_size() \
                                  else clst_v, clst_u 
        big.merge(sml)
        self._loc_cluster_map.pop(sml)

    def grow_vertex(self, vertex: Vertex) -> Tuple:
        """Given a vertex to grow, returns a list of potential fusion edges and
        a list of potential new boundary vertices

        Parameters
        ----------
        vertex : Vertex
            The vertex to be grown

        Returns
        -------
        fusion_list : List[Edge]
            The list of new potential fusion edges.

        new_boundary : List[Vertex]
            The list of new potential boundary vertices.
        """
        new_boundary: List[Vertex] =[]
        fusion_list: List[Edge] = []
        surround_edges_loc: List[Tuple] = self._get_surrond_edges(vertex)
        for e in surround_edges_loc:
            edge = self._edge_newly_grown(e)
            if edge: # if edge is growing to be grown
                if vertex is edge.fst:
                    # append new vertex, but check if it has been approached 
                    # from the otehr end
                    loc_u = self._get_other_vertex_loc(vertex, edge)
                    if self._status[loc_u] == Support.DARK_POINT:
                        vertex_u = self._light_up_vertex(loc_u, vertex)#11
                        new_boundary.append(vertex_u)
                        edge.add_vertex(snd = vertex_u) #11
                    else:
                        # but probably the same cluster
                        vertex_u = self._loc_vertex_map[loc_u]#11
                        edge.add_vertex(snd = vertex_u) #11
                        fusion_list.append(edge)
                else:
                    # but probably the same cluster
                    edge.add_vertex(snd = vertex)
                    fusion_list.append(edge)

        return (fusion_list, new_boundary)
    
    def _edge_newly_grown(self, loc: Tuple, v: Vertex) -> Edge:
        """Given the coordinate of an edge, return the edge instance if the edge 
        becomes grown from half-grown, returns None otherwise.

        Parameters
        ----------
        loc : Tuple
            The coordinates of the edge to be grown.
        
        v : Vertex
            The vertex that the edge grows from.

        Returns
        -------
        edge : Edge
            The edge instance if the edge becomes grown from half-grown, 
            None otherwise.
        """
        status = self._status[loc]

        if status == Support.HALF_GROWN:
            edge = self._loc_edge_map[loc]
            self._status[loc] = Support.GROWN
            return edge
        elif status == Support.UNOCCUPIED:
            edge = Edge(loc)
            edge.add_vertex(fst=v)
            self._loc_edge_map[loc] = edge
            self._status[loc] = Support.HALF_GROWN
        
        return None

    def _light_up_vertex(self, loc: Tuple, parent: Vertex):
        """Given the coordinate and parent vertex of the new vertex, returns 
        the new Vertex instance of the given coordinates.

        Parameters
        ----------
        loc : Tuple
            The coordinates of the vertex to be lit

        parent : Vertex
            The parent of the new vertex

        Returns
        -------
        vertex : Vertex
            The vertex instance of the newly created vertex.
        """
        vertex = Vertex(loc, parent=parent)
        self._status[loc] = Support.VERTEX
        self._loc_vertex_map[loc] = vertex
        root = parent.add_child(vertex)
        self._loc_cluster_map[root.get_location()].size_increment()
        return vertex    

    def _get_other_vertex_loc(self, vertex: Vertex, edge: Edge) -> Tuple:
        """Given a vertex and an edge of the vertex, returns the coordinates of
        the vertex at the other end of the edge.

        Parameters
        ----------
        vertex : Vertex
            The vertex representaion

        edge : Edge
            The edge attached to the given vertex

        Returns
        -------
        (x, y) : Tuple
            The coordinates of the vertex at the other end of the edge.
        """
        (v_x, v_y) = vertex.get_location()
        (e_x, e_y) = edge.get_location()

        (x, y) = (0, 0)
        if v_x == e_x:
            (x, y) = (v_x, (e_y + (e_y - v_y)) % self._L_y) 
            # true because vertex is on the other side
        elif v_y == e_y:
            (x, y) = ((e_x + (e_x - v_x)) % self._L_x, v_y)
        
        return (x, y)

    
    def _get_surrond_edges(self, vertex: Vertex) -> List[Tuple]:
        """Given a vertex, returns the coordinates of its surrounding edges.

        Parameters
        ----------
        vertex : Vertex
            the vertex representation

        Returns
        -------
        edges : List[Tuple]
            The coordinates list of the surrounding edges.
        """
        (x, y) = vertex.get_location()
        edges = [((x - 1) % self._L_x, y),
                 (x, (y - 1) % self._L_y),
                 ((x + 1) % self._L_x, y),
                 (x, (y + 1) % self._L_y)]
        return edges

