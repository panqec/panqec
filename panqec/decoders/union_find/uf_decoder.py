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


        ### Peeler decoder -- Z error ###

        ## first we construct lattice/grid graphs  for each cluster.-- to do later##
        # these graphs are stored as an adjacency list that repeats the edges (since it stores the neighbours of each node)
        # note that if the edges are note repeated this will affect the edge removal in the tree finding function below (then need to add an
        # extra if function). Note that in each graph the nodes are represented by integer indices from 0 to n-1 (for n nodes) in this order.



        ## now assuming we have the cluster graphs we find spanning forest ##
        def nu_nodes(a):
            """Getting the number of nodes of a graph from

            Input:  a  Dictionary representing the adjaceny list of the graph
            Output:    Integer representing the number of nodes of the graph                
            """
            return len(a.keys())

        def find_neighbours(g,n):
            """Function that finds the neighbour of a node within a graph.

            Inputs:
            g the dictionary representing the adjacency list of the graph
            n the node whose neighbours are to be found

            Output:
            l a list of the neighbour (indices) of the node n            
            """
            return g[n]

        def find_tree(g, s=0):
                """Function that finds a spanning tree for a graph
                Input: 
                g  dictionary representing the adjacency list of the graph for which the spanning tree is to be found
                s  the node to start the BFS algorithm at. 0 by default

                Output:
                tree the adjaceny list of the spanning tree of g                
                
                """

                q = [] # empty list representing the que
                q.append(s) # adding s to the que

                n = nu_nodes(g) # number of nodes of g
                visited = [False]*n # array to check if a node has been visited
                visited[s]= True # mark start node as being visited

                tree = g # initialise tree adjacency list

                while len(q)>0 : # while the que is not empty
                    node = q[0] # the node at the start of the que
                    q.pop(0)  # remove "node" from the que

                    for next in find_neighbours(g,node): # looping through the neighbours of "node"

                        if !visitied[next]:
                            q.append(next)
                            visited[next] = True                       

                            
                        elif (next in tree[node]): # check if the edge between "next" and "node" exists
                            # remove the edge between "next" and "node" entirely from the tree graph
                            tree[node].remove(next)
                            tree[next].remove(node)

                return tree


                
        ## Assuming L is a list/array of the individual cluster graphs (ie list/array of dictionaries), we find the forest:

        forest = [] # initialise the forest list
        for graph in L:
            tree = find_tree(graph)
            forest.append(tree)            
            

        ## Now the peeling the forest ##

        ## assuimung we make a list of syndromes within each cluster with the vertices names 0 to n, called sig_i such that we have all
        ## syndromes as sig = list of all sig_i. Alternatively fix the indexing of the cluster graphs to be 0 to N, where N is the number of 
        ## nodes in all clusters. If you adopt this indexing, then (maybe) change the way the nodes are indexed in the dictionaries to become 
        ## the key name ++ other changes needed!.

        ## performing the peeler


        A = []  # list of edges to be corrected
        for i,tree in enumerate(forest):
            sig_i = sig[i] # pick the syndromes list for the appropriate cluster
            leaves = [] # list of leaf nodes (pendant vertices)
            for node in tree.keys(): # append node to "leaves" if it has one neighbour
                if len(tree[node])==1: 
                    leaves.append(node)                     

            while len(tree)>0: # while tree is not empty (not fully peeled)
                v = leaves[0] # pendant vertex chosen randomly (first element of "leaves")
                u = tree[v][0] # neighbour of the pendant vertex

                ## remove leaf v from "leaves" and from the tree.
                leaves.remove(v) # remove v from leaves
                tree.pop(v) # remove v from the tree
                tree[u].remove(v) # remove v from the neighbours list of u

                ## check if u is a pendant vertex now, and if so add to leaves
                if len(tree[u])==1:
                    leaves.append(u)

                ## peeler algorithm

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
