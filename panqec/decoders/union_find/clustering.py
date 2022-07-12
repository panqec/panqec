from abc import ABCMeta, abstractmethod
from __future__ import annotations
from msilib.schema import Error
from os import stat
from re import U
from typing import Any, Dict, Tuple, List
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
        for c in odd_clusters:
            fusion_list += c.grow()
            # fusion_list.append(grow(cluster_forest(u))) #inlcude step 9
        
        for e in fusion_list:
            if e.fst.find_root() is not e.snd.find_root():
                union(e.u, e.v) #include step 7,8
            else:
                fusion_list.remove(e)
        
        update_verices(vertices) #remove even cluster       

    return #grwon supprt and syndrome


def union(vt_v, vt_u, forest):
    #TODO
    return

class Cluster_Tree():

    def __init__(self, root):
        self._size = 1
        self._odd = True
        self._root: Vertex = root
        self._boundary_list: List[Vertex] = [root]
    
    def is_odd(self) -> Boolean:
        return self._odd
    
    def size_increment(self, inc = 1):
        self._size += inc
    
    def grow(self, support: Support):
        """ 
            grow the boundary list of the cluster, returns fusion list 
        """
        #TODO update boundary list
        return map(lambda v : v.grow(support) , self._boundary_list)

        
        

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
        #TODO path compression
        v = self
        p = v.get_parent()
        while p is not None:
            v = p
            p = v.get_parent()
        return v
    
    def grow(self, support: Support):
        #TODO move support grow to here
        support.grow(self)

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
        self._loc_cluster_map: Dict[Tuple, Cluster_Tree] = {}
        self._loc_edge_map = {}

        self._init_loc_syndrome(syndrome)
        self._init_loc_cluster(syndrome)
    
    def _init_loc_syndrome(self, syndrome: List[T]) -> Dict[T, Vertex]:
        self._status[syndrome] = Support.SYNDROME  # light up the seen vertex
        return dict(map(lambda l : (l, Vertex(l)), syndrome))
    
    def _init_loc_cluster(self, locations: List[T]) -> Dict[T, Cluster_Tree]:
        return dict(map(lambda l : (l, Cluster_Tree(self._loc_vertex_map[l])), 
                        locations))
    
    def get_clusters(self) -> List:
        return list(self._loc_cluster_map.value())

    def grow(self, vertex: Vertex):
        #TODO refactor
        fusion_list: List[Edge] = []
        surround_edges_loc: List[Tuple] = self._get_surrond_edges(vertex)
        for e in surround_edges_loc:
            status = self._status[e]
            
            if status == Support.UNOCCUPIED:
                edge = Edge(e)
                edge.add_vertex(vertex)
                self._loc_edge_map[e] = edge
                self._status[e] = Support.HALF_GROWN
                grown = False
            elif status == Support.HALF_GROWN:
                edge = self._loc_edge_map[e]
                self._status[e] = Support.GROWN
                if vertex is edge.fst:
                    # append new vertex, but check if it has been approached 
                    # from the otehr end
                    loc_u = self._get_other_vertex_loc(vertex, edge)
                    u_status = self._status[loc_u]
                    if u_status == Support.DARK_POINT:
                        vertex_u = Vertex(loc_u, parent=vertex)
                        #refactor
                        self._status[loc_u] = Support.VERTEX
                        self._loc_vertex_map[loc_u] = vertex_u
                        #
                        root = vertex.add_child(vertex_u) #TODO cluster size++
                        self._loc_cluster_map[root.get_location()].size_increment()
                        edge.add_vertex(snd = vertex_u)

                    else:
                        # but probably the same cluster
                        edge.add_vertex(snd = vertex)
                        fusion_list.append(edge)
                else:
                    edge.add_vertex(snd = vertex)
                    fusion_list.append(edge)

        return fusion_list
    
    def _get_other_vertex_loc(vertex: Vertex, edge: Edge):
        """
            Given an edge and a vertex attached, infer the coordinate of 
            the other vertex of the edge.
        """
        #TODO overflow bound
        (v_x, v_y) = vertex.get_location
        (e_x, e_y) = edge.get_location

        (x, y) = (0, 0)
        if v_x == e_x:
            (x, y) = (v_x, e_y + (e_y - v_y)) 
            # true because vertex is on the other side
        elif v_y == e_y:
            (x, y) = (e_x + (e_x - v_x), v_y)
        
        return (x, y)

    
    def _get_surrond_edges(self, vertex: Vertex) -> List[Tuple]:
        """
        Input: vertex to extract for its coordinates

        Return: the edges coordinate around the vertex.
        
        """

        #TODO get around the graph
        (x, y) = vertex.get_location()
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

