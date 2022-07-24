from abc import ABCMeta, abstractmethod
from __future__ import annotations
from msilib.schema import Error
from os import stat
from re import U
from typing import Any, Dict, Set, Tuple, List
from xmlrpc.client import Boolean
import numpy as np
from pyrsistent import T, b, s

def clustering(syndrome):
    # vertices: List = transfer_syndrome(syndrome) # turn syndromes into vertices
    # cluster_forest = map_vertex_tree(vertices) # map vertices with cluster information
    support = Support(syndrome)

    odd_clusters = filter(lambda c : c.is_odd(), support.get_clusters())

    while odd_clusters:
        fusion_list = []
        for c in odd_clusters: # should only update the smallest cluster, no loop
            fusion_list += c.grow()
            # fusion_list.append(grow(cluster_forest(u))) #inlcude step 9
        
        for e in fusion_list:
            if e.fst.find_root() is not e.snd.find_root():
                support.union(e.fst, e.snd) #include step 7,8
            # else:
            #     fusion_list.remove(e)
        #TODO 111 update THE cluster
        for c in odd_clusters: 
            c.update_boundary(support)

        odd_clusters = filter(lambda c : c.is_odd(), support.get_clusters())

    return #grwon supprt and syndrome

class Cluster_Tree():

    def __init__(self, root):
        self._size = 1
        self._odd = True
        self._root: Vertex = root
        self._boundary_list: Set[Vertex] = [root]
    
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
        """ 
            grow the boundary list of the cluster, returns fusion list 
        """
        #TODO update boundary list
        new_boundary = []
        fusion_list = []
        for v in self._boundary_list:
            # what if no boundary list?
            (fl, nb) = support.grow_vertex(v)
            new_boundary += nb
            fusion_list += fl
        
        self._boundary_list.union(new_boundary)
        return fusion_list
    
    def merge(self, clst: Cluster_Tree):
        self._size += clst.get_size()
        self._odd = False
        rt = clst.get_root()
        rt.set_parent(self._root)
        self._root.add_child(rt)
        self._boundary_list.union(clst.get_boundary())
        
    def update_boundary(self, support: Support):
        self._boundary_list = set(filter(lambda b : support.vertex_is_boundary(b),
                                     self._boundary_list))

class Vertex():

    def __init__(self,
                 location: Tuple,
                 parent: Vertex = None,
                 children: List[Vertex] = []):
        self._location = location
        self._parent = parent
        self._children = children
    
    def get_location(self):
        return self._location
    
    def add_child(self, child):
        """
            Add new children, return root for cluster size increase.
        """
        self._children.append(child)
        return self.find_root()
    
    def remove_child(self, child):
        self._children.remove(child)

    def set_parent(self, new_parent):
        self._parent = new_parent
    
    def get_parent(self):
        return self._parent

    def find_root(self) -> Vertex:
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
            r.add_child(v)
            v.set_parent(r)


class Edge():

    def __init__(self, location):
        self.fst: Vertex = None
        self.snd: Vertex = None
        self._location = location
    
    def get_location(self) -> Tuple:
        return self._location

    def add_vertex(self, fst=None, snd=None):
        self.fst = fst
        self.snd = snd


class Support():
    # edges status
    UNOCCUPIED = 0
    HALF_GROWN = 1
    GROWN      = 2

    # vertex status
    DARK_POINT = 0
    VERTEX     = 1
    SYNDROME   = 2
    # improvement1: Eliminate faces...
    # improvement2: Eliminate vertex from the support matrix, need hash function

    def __init__(self, syndrome: List[Tuple], x_len, y_len):
        self._x_len = x_len
        self._y_len = y_len
        self._status = np.zeros((x_len, y_len), dtype='uint8')
        self._loc_vertex_map = {}
        self._loc_cluster_map: Dict[Tuple, Cluster_Tree] = {} # root loc map cluster
        self._loc_edge_map = {}

        self._loc_vertex_map = self._init_loc_syndrome(syndrome)
        self._loc_cluster_map = self._init_loc_cluster(syndrome)
    
    def _init_loc_syndrome(self, syndrome: List[T]) -> Dict[T, Vertex]:
        self._status[syndrome] = Support.SYNDROME  # light up the seen vertex
        return dict(map(lambda l : (l, Vertex(l)), syndrome))
    
    def _init_loc_cluster(self, locations: List[T]) -> Dict[T, Cluster_Tree]:
        return dict(map(lambda l : (l, Cluster_Tree(self._loc_vertex_map[l])), 
                        locations))
    
    def get_clusters(self) -> List:
        return list(self._loc_cluster_map.value())

    def vertex_is_boundary(self, v: Vertex):
        return [] != \
            filter(lambda s : s != Support.GROWN, \
                self._status[self._get_surrond_edges(v)])

    def union(self, rt_v: Vertex, rt_u: Vertex):
        clst_v = self._loc_cluster_map[rt_v.get_location()]
        clst_u = self._loc_cluster_map[rt_u.get_location()]

        big, sml = clst_u, clst_v if clst_u.get_size() > clst_v.get_size() \
                                  else clst_v, clst_u 
        big.merge(sml)
        self._loc_cluster_map.pop(sml)

    def grow_vertex(self, vertex: Vertex):
        #TODO refactor
        new_boundary: List[Vertex] =[]
        fusion_list: List[Edge] = []
        surround_edges_loc: List[Tuple] = self._get_surrond_edges(vertex)
        for e in surround_edges_loc:
            status = self._status[e]
            
            if status == Support.UNOCCUPIED:
                edge = Edge(e) #1
                edge.add_vertex(vertex) #1
                self._loc_edge_map[e] = edge #1
                self._status[e] = Support.HALF_GROWN #1
            elif status == Support.HALF_GROWN:
                edge = self._loc_edge_map[e] #1
                self._status[e] = Support.GROWN #1
                if vertex is edge.fst:
                    # append new vertex, but check if it has been approached 
                    # from the otehr end
                    loc_u = self._get_other_vertex_loc(vertex, edge)
                    if self._status[loc_u] == Support.DARK_POINT:
                        vertex_u = Vertex(loc_u, parent=vertex)
                        # ^^ new boundary
                        new_boundary.append(vertex_u)
                        #refactor
                        self._status[loc_u] = Support.VERTEX
                        self._loc_vertex_map[loc_u] = vertex_u
                        #
                        root = vertex.add_child(vertex_u)
                        self._loc_cluster_map[root.get_location()].size_increment()
                        edge.add_vertex(snd = vertex_u)

                    else:
                        # but probably the same cluster
                        edge.add_vertex(snd = vertex)
                        fusion_list.append(edge)
                else:
                    edge.add_vertex(snd = vertex)
                    fusion_list.append(edge)

        return (fusion_list, new_boundary)
    
    def _get_other_vertex_loc(self, vertex: Vertex, edge: Edge):
        """
            Given an edge and a vertex attached, infer the coordinate of 
            the other vertex of the edge.
        """
        (v_x, v_y) = vertex.get_location()
        (e_x, e_y) = edge.get_location()

        (x, y) = (0, 0)
        if v_x == e_x:
            (x, y) = (v_x, (e_y + (e_y - v_y)) % self._y_len) 
            # true because vertex is on the other side
        elif v_y == e_y:
            (x, y) = ((e_x + (e_x - v_x)) % self._x_len, v_y)
        
        return (x, y)

    
    def _get_surrond_edges(self, vertex: Vertex) -> List[Tuple]:
        """
        Input: vertex to extract for its coordinates

        Return: the edges coordinate around the vertex.
        
        """
        (x, y) = vertex.get_location()
        edges = [((x - 1) % self._x_len, y),
                 (x, (y - 1) % self._y_len),
                 ((x + 1) % self._x_len, y),
                 (x, (y + 1) % self._y_len)]
        return edges

