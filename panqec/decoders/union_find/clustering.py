from abc import ABCMeta, abstractmethod
from typing import Tuple, List



class Cluster_Tree():

    def __init__(self):
        self._size_ = 1
        self._root_ = None

class Vertex(metaclass=ABCMeta):

    def __init__(self,
                 location: Tuple,
                 parent=None,
                 children=[]):
        self._location_ = location
        self._parent = parent
        self._children= children
        
    

class Root(Vertex):

    def __init__(self, 
                 location: Tuple, 
                 cluster_tree: Cluster_Tree,
                 parent=None, 
                 children=[]):
        super().__init__(location)
        self._cluster_tree = cluster_tree

class Node(Vertex):
    def __init__(self, 
                 location: Tuple,
                 parent=None, 
                 children=[]):
        super().__init__(location, parent, children)