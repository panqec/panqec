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
        # and we have Hz as the parity matrix for z stabilizers (a 2D numpy array)

        ### Peeler decoder -- Z error ###

        ## constructing the graphs and the syndromes of each cluster ##

        # Note that is is very expensive when it comes to the complexity

        # These graphs are stored as an adjacency list that repeats the edges (since it stores the neighbours of each node), ie the
        # adjacency list format used: {0:[1,2], 1:[0,2], 2:[0,1]} ; 0,1,2 are the indices of the nodes.        
        # Note 1) if the edges are not repeated this will affect the edge removal in the tree finding function below (then need to add an
        # extra if function).
        # Note 2) in each graph the nodes (lattice vertices) are represented by their panqec integer indices. 


        graphs = [] # list of all graphs organized per cluster in the form of a list of dictionaries
        sig = [] # list of all syndromes organized per cluster in the form of a list of lists

        for cluster in output:           
            graph_i = {} # initialize the graph dictionary of the ith cluster
            sig_i = [] # initializing the list of syndromes of the ith cluster

            ## creating the graphs:

            vertices_i = cluster[0] # list of vertices (nodes) in the ith cluster
            edges_i = cluster[1]  # list of qubits (edges) in the ith cluster            

            # looping through all the nodes in the cluster to find their neighbouring nodes and create an adjaceny list
            # for the graph representing the cluster. The problem here is we double the search as neighbours are recorded
            # twice in the adjacency list.
            for vertex in vertices_i: 
                
                # finding list of qubits/edges in the cluster attached to the node "vertex" using the parity check matrix
                qubits_list = [] # can have a size of 1-4
                for edge in edges_i: #--> np.where can be used
                    if Hz[vertex,edge] == 1:
                        qubits_list.append(edge)

                # for each such qubit/edge, we find which other node from the cluster is attached to it. These
                # other nodes are the neighbours of the node "vertex"
                neighbours = []
                for qubit in qubits_list:
                    # we don't want to consider the node "vertex" as its own neighbour, but instead of using an if statement
                    # it is faster to just remove "vertex" from the vertices_i --> if is better!!!
                    vertices_list = vertices_i[:] # making a duplicate of the list of vertices in the cluster --> not needed if u use "if"
                    vertices_list.remove(vertex) # removing "vertex" 
                    for vertex_ in vertices_list: # we look at only the nodes already in the cluster discluding "vertex"--> np.where can be used                      
                        if Hz[vertex_, qubit] == 1:
                            neighbours.append(vertex_)                              
                
                graph_i[vertex] = neighbours

                ## organizing the syndromes per graph
                if vertex in syndrome:
                    sig_i.append(vertex)

            sig.append(sig_i)
            graphs.append(graph_i)



        ## Now we find spanning forest ##

        ## Functions that find the spanning tree of a graph:

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
                parent = visited # a dictionary to check if a node is a parent node --> make sure to duplicate
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
                                tree[current_node].remove(next) #--> maybe find another way but it is only upto 4 so ok
                                tree[next].remove(current_node) # --> find another way but only 4 so ok

                return tree

           
                
        ## Finding the forest:

        forest = [] # initialise the forest list
        for graph in graphs:
            tree = find_tree(graph)
            forest.append(tree)            
            

        ## Now peeling the forest and finding the correction ##


        ## performing the peeler

        A_ = []  # list of edges to be corrected in the form of a tuple of the bounding vertices of that edge
        for i,tree in enumerate(forest):
            sig_i = sig[i] # pick the syndromes list for the appropriate cluster
            leaves = [] # list of leaf nodes, the pendant vertices
            for node in tree.keys(): # append node to "leaves" if it has one neighbour
                if len(tree[node])==1: 
                    leaves.append(node)                     

            while len(tree.keys()): # while tree is not empty 
                v = leaves[0] # choose pendant vertex to work with randomly (first element of "leaves")
                u = tree[v][0] # neighbour of the pendant vertix, which is the vertix connecting the leaf edge to the forest

                # removing leaf edge from "leaves" and from the tree.
                leaves.pop(0) # remove v from leaves --> v is leaf node 0 so we can use pop
                tree.pop(v) # remove v from the tree --> not good for complexity? for dictionary it is of order of one access time so ok
                tree[u].remove(v) # remove v from the neighbours list of u --> upto 4 elements in the list

                # checking if u is a pendant vertex now, and if so add to leaves
                if len(tree[u])==1:
                    leaves.append(u)

                # peeler algorithm            
                if u in sig_i: 
                    e = (u,v) # edge "e" to be corrected stored as a tuple e = (v1,v2) where v1 and v2 are the vertices bounding "e"
                    A_.append(e)  
                    sig_i.remove(u) 
                    if v in sig_i:
                        sig_i.remove(v)
                    else:
                        sig_i.append(v)




        ## Now proccess A_ to find Ex of the bsf ##

        ## Processing A_ to extract the edge/qubit indices in panqec --> Hz * Hz^T  or use panqec coordinates ( the average of the coordinates of v1 & v2, this can be a function) 

        # note that is could be expensive when it comes to the complexity
        
        A = [] # list of indices for the qubits to be corrected
        # for each edge in A_, we loop through all the edges in Hz to see which edge is attached to the two vertices v1 and v2
        # in the tuple e = (v1,v2). Attached means we have a 1 in the Hz matrix.   
        for edge in A_:
            v1 = edge[0]
            v2 = edge[1]
            N =  len(Hz[0]) # total number of qubits in the code
            for qubit in range(N): # --> 
                if (Hz[v1,qubit]==1) and (Hz[v2,qubit]==1): # if "qubit" attached to both v1 and v2
                    A.append(qubit)



        ## Finding Ex
        correction_x= [0]*N # np.zeros(N)
        for qubit in A: # we find Ex by having 1 for the qubits that are to be corrected and 0 for the other qubits
            Ex[qubit] = 1

        
        ### Load the correction into the X block of the full bsf ###
        correction[:self.code.n] = correction_x
        # correction[self.code.n:] = correction_z

        return correction


