import numpy as np
from pymatching import Matching
from panqec.decoders import BaseDecoder
from panqec.codes import Toric2DCode
from panqec.error_models import BaseErrorModel


class UnionFindDecoder(BaseDecoder):
    """Union Find decoder for 2D Toric Code"""

    label = 'Toric 2D Union Find'

    def __init__(self,
                 code: Toric2DCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        super().__init__(code, error_model, error_rate)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)

        syndromes_z = self.code.extract_z_syndrome(syndrome)
        syndromes_x = self.code.extract_x_syndrome(syndrome)

        print("We are decoding with union find!!!")

        # TODO

        ### clustering/syndrome validation stage. Outputs cluster trees and support -- Lynna


        #assume the output is output = [1:(s[0,1,2], q[3,4,5]), 2:(s[22,33,44], q[34,46,57]), ...]
        # and syndrome = [2,3,4] which is a list of all vertix indices that are syndromes

        ### Peeler decoder -- Z error ###

        ## first we construct lattice/grid graphs  for each cluster ##
        # these graphs are stored as an adjacency list that repeats the edges (since it stores the neighbours of each node)
        # note that if the edges are note repeated this will affect the edge removal in the tree finding function below (then need to add an
        # extra if function). Note that in each graph the nodes (lattice vertices) are represented by their panqec integer indices 
        # in the keys of the dictionary, ie Adjacency list format used: {0:[1,2], 1:[0,2], 2:[0,1]} ; 0,1,2 are the indices of the nodes


        # constructing the graphs --> Lynna

        graphs = # the final output which is a list of dictionaries

        # the syndromes of each graph as a list of lists called "sig"
        sig = [] # list of all syndromes organised by cluster
        for graph in output:
            vertices_i = graph[0]
            sig_i = []
            for vertix in vertices_i:
                if vertix in syndrome:
                    sig_i.append(vertix)
            sig.append(sig_i)









        ## now assuming we have the cluster graphs we find spanning forest ##
        def nu_nodes(g):
            """Getting the number of nodes of a graph from its adjacency list

            Input:  g  Dictionary representing the adjaceny list of the graph
            Output:    Integer representing the number of nodes of the graph                
            """
            return len(g.keys())

        def find_neighbours(g,n):
            """Function that finds the neighbour of a node within a graph.

            Inputs:
            g the dictionary representing the adjacency list of the graph
            n the index of the node whose neighbours are to be found

            Output:
            a list of the indices of the neighbours of the node n            
            """
            return g[n]

        def find_tree(g):
                """Function that finds a spanning tree for a graph g. This function uses a BFS algorithm and starts at the first
                node in the adjaceny dictionary of g (first node means the first key of the dictionary).
                Input: 
                g      dictionary representing the adjacency list of the graph for which the spanning tree is to be found
                
                Output:
                tree   the adjaceny list of the spanning tree of g                
                
                """
                tree = g # initialise tree adjacency list
                nodes = list(tree.keys()) # list of nodes in the tree

                q = [] # empty list representing the que
                s = nodes[0] # the first key (node) of the tree dictionary
                q.append(s) # adding s to the que

                n = nu_nodes(g) # number of nodes of g

                # creating "visited" and "parent" dictionaries which are both initialised as False for all the nodes
                values = [False]*n                 
                visited = {nodes[i]:values[i] for i in range (n)} # a dictionary to check if a node has been visited
                parent = visited # a dictionary to check if a node is a parent node
                visited[s]= True # mark start node as being visited
                
                while len(q): # while the que is not empty
                    current_node = q[0] # the node at the start of the que
                    q.pop(0)  # remove "current_node" from the que
                    parent[current_node] = True # "current_node" is now a parent node

                    for next in find_neighbours(tree,current_node): # looping through the neighbours of "current_node" in the tree

                        if not parent[next]: # if a parent node we do nothing
                        
                            if not visited[next]: # if node was not visited before, add to the que
                                q.append(next)
                                visited[next] = True                       

                                
                            else: # if node is visited
                                # remove the edge between "next" and "current_node" entirely from the tree graph
                                tree[current_node].remove(next)
                                tree[next].remove(current_node)

                return tree


                
        ## Assuming graphs is a list/array of the individual cluster graphs (ie list/array of dictionaries), we find the forest:

        forest = [] # initialise the forest list
        for graph in graphs:
            tree = find_tree(graph)
            forest.append(tree)            
            

        ## Now peeling the forest ##

        ## we assume that we  have a list of syndromes within each cluster with the vertices named 0 to n, called sig_i such that we have all
        ## syndromes as sig = list of all sig_i. Alternatively fix the indexing of the cluster graphs to be 0 to N, where N is the number of 
        ## nodes in all clusters. If you adopt this indexing, then (maybe) change the way the nodes are indexed in the dictionaries to become 
        ## the key name ++ other changes needed!.

        ## performing the peeler


        A = []  # list of edges to be corrected
        for i,tree in enumerate(forest):
            sig_i = sig[i] # pick the syndromes list for the appropriate cluster
            leaves = [] # list of leaf nodes, the pendant vertices
            for node in tree.keys(): # append node to "leaves" if it has one neighbour
                if len(tree[node])==1: 
                    leaves.append(node)                     

            while len(tree.keys()): # while tree is not empty 
                v = leaves[0] # choose pendant vertex to work with randomly (first element of "leaves")
                u = tree[v][0] # neighbour of the pendant vertix, which is the vertix connecting the leaf edge to the forest

                ## remove leaf edge from "leaves" and from the tree.
                leaves.remove(v) # remove v from leaves
                tree.pop(v) # remove v from the tree
                tree[u].remove(v) # remove v from the neighbours list of u

                ## check if u is a pendant vertex now, and if so add to leaves
                if len(tree[u])==1:
                    leaves.append(u)

                ## peeler algorithm
                # this requires sig_i to have the cluster indices of the syndrome vertics
                if u in sig_i: 
                    e = (u,v) # edge to be corrected
                    A.append(e)  
                    sig_i.remove(u) 
                    if v in sig_i:
                        sig_i.remove(v)
                    else:
                        sig_i.append(v)


        ## now proccess A to extract the edge indices in panqec
        ### Matching the Panqec syntax ###

        
        # Load it into the X block of the full bsf.
        # correction[:self.code.n] = correction_x
        # correction[self.code.n:] = correction_z

        return correction
