from abc import ABCMeta, abstractmethod
from __future__ import annotations
from typing import Tuple, List
import numpy as np
from pyrsistent import s

def clustering(syndrome):
    vertices: List = transfer_syndrome(syndrome) # turn syndromes into vertices
    cluster_forest = map_vertex_tree(vertices) # map vertices with cluster information
    support = Support()

    while vertices:
        fusion_list = []
        for u in vertices:
            fursion_list = grow(cluster_forest(u)) #inlcude step 9
        
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
        self._size_ = 1
        self._odd_ = True
        self._root_: Vertex = root
        self._boundary_list_: List[Vertex] = [root]

class Vertex():

    def __init__(self,
                 location: Tuple,
                 parent: Vertex = None,
                 children: List[Vertex] = []):
        self._location_ = location
        self._parent_ = parent
        self._children_ = children
    
    def add_child(self, child):
        self._children_.append(child)
    
    def remove_child(self, child):
        self._children_.remove(child)

    def set_parent(self, new_parent):
        self._parent_ = new_parent
    
    def get_parent(self):
        return self._parent_

    def find(self) -> Vertex:
        pass

def Support():
    UNOCCUPIED = 0
    HALF_GROWN = 1
    GROWN      = 2

    def __init__(self, x_len, y_len):
        self._x_len_ = x_len
        self._y_len_ = y_len
        self._edges_ = np.zeros((x_len, y_len), dtype='uint8')

    def grow(vertex):
        return

def map_vertex_tree(vertices: List[Vertex]):
    return dict(map(lambda v : (v, Cluster_Tree(v)), vertices))