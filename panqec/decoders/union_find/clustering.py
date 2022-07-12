from abc import ABCMeta, abstractmethod
from __future__ import annotations
from os import stat
from re import U
from typing import Any, Dict, Tuple, List
from xmlrpc.client import Boolean
import numpy as np
from pyrsistent import T, b, s

def clustering(syndrome):
    vertices: List = transfer_syndrome(syndrome) # turn syndromes into vertices
    cluster_forest = map_vertex_tree(vertices) # map vertices with cluster information
    support = Support(syndrome)

    while vertices:
        fusion_list = []
        for u in vertices:
            fusion_list.append(grow(cluster_forest(u))) #inlcude step 9
        
        for e in fusion_list:
            if find(e.u) != find(e.v):
                union(e.u, e.v) #include step 7,8
            else:
                fusion_list.remove(e)
        
        update_verices(vertices) #remove if true       

    return #grwon supprt and syndrome


def union(vt_v, vt_u, forest):
    return

class Cluster_Tree():

    def __init__(self, root):
        self._size = 1
        self._odd = True
        self._root: Vertex = root
        self._boundary_list: List[Vertex] = [root]
    
    def is_odd(self) -> Boolean:
        return self._odd
        

class Vertex():

    def __init__(self,
                 location: Tuple,
                 parent: Vertex = None,
                 children: List[Vertex] = []):
        self._location = location
        self._parent = parent
        self._children = children
    
    def add_child(self, child):
        self._children.append(child)
    
    def remove_child(self, child):
        self._children.remove(child)

    def set_parent(self, new_parent):
        self._parent = new_parent
    
    def get_parent(self):
        return self._parent

    def find_root(self) -> Vertex:
        v = self
        p = v.get_parent()
        while p is not None:
            v = p
            p = v.get_parent()
        return v

class Edge():
    

    def __init__(self, location):
        self.fst: Vertex = None
        self.snd: Vertex = None
        self._location = location
    
    def add_vertex(self, fst=None, snd=None):
        self.fst = fst
        self.snd = snd


def Support():
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
        self._loc_cluster_map = {}
        self._loc_edge_map = {}

        _init_loc_syndrome(syndrome)
        _init_loc_cluster(syndrome)
    
    def _init_loc_syndrome(self, syndrome: List[T]) -> Dict[T, Vertex]:
        self._status[syndrome] = SYNDROME  # light up the seen vertex
        return dict(map(lambda l : (l, Vertex(l)), syndrome))
    
    def _init_loc_cluster(self, locations: List[T]) -> Dict[T, Cluster_Tree]:
        return dict(map(lambda l : (l, Cluster_Tree(self._loc_vertex_map[l])), 
                        locations))

    def grow(self, vertex: Vertex, fusion_list: List[Edge]):
        surround_edges_loc: List[Tuple] = _get_surrond_edges(vertex)
        new_grown =[]
        grown = True
        for e in surround_edges_loc:
            status = self._status[e]
            
            if status is UNOCCUPIED:
                edge = Edge(e)
                edge.add_vertex(vertex)
                self._loc_edge_map[e] = edge
                self._status[e] = HALF_GROWN
                grown = False
            elif status is HALF_GROWN:
                edge = self._loc_edge_map[e]
                if vertex is edge.fst:
                    # append new vertex, but check if it has been approached 
                    # from the otehr end
                    if 
                else:
                    edge = self._loc_edge_map[e]
                    edge.add(snd = vertex)
                    self._status[e] = GROWN
                    fusion_list.append(edge)







        #TODO Next
        return
    
    def _get_surrond_edges(self, vertex: Vertex) -> List[Tuple]:
        """
        Input: vertex to extract for its coordinates

        Return: the edges coordinate around the vertex.
        
        """

        #TODO get around the graph
        (x, y) = vertex.location
        edges = []
        if x > 0:
            edges.append((x - 1, y))
        if y > 0:
            edges.append((x, y - 1))
        if x + 1 < self._x_len:
            edges.append((x + 1, y))
        if y + 1 < self._y_len:
            edges.append((x, y + 1))
        
        return edges




def grow(tree: Cluster_Tree):

