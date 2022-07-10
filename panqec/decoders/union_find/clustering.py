from abc import ABCMeta, abstractmethod
from __future__ import annotations
from typing import Tuple, List

from pyrsistent import s



class Cluster_Tree():

    def __init__(self):
        self._size_ = 1
        self._root_: Vertex = None
        self._boundary_list: List[Vertex] = []

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
