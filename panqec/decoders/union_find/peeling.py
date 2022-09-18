import numpy as np
import copy

def peeling(output, syndromes, H):
    """
    A function that takes in the output from the clustering function (in panqec.decoders.union_find.clustering),
    which is a tuple containing a list of stabiliser indices (first element) and a list of qubit indices (second element)
    which are for each cluster, to create a spanning forest and peel it to obtain a correction for one type of error.

    Parameters
    ----------
    output: np.ndarray
        The output of the clustering function from An array of tuples of the formpanqec.decoders.union_find.clustering
        This is an array of tuples of the form [([],[]), ([],[]) ..]. Each tuple holds the information of one of the 
        clusters generated. The tuples are such that the first element is a list of stabilizers (indices) and the second
        element is a list of qubits (indices) for that particular cluster.

   syndromes: np.ndarray
        Syndrome as an array of size m, where m is the number of stabilizers. Each element contains 1 if the stabilizer is
        activated and 0 otherwise. This is the syndromes for one type of error
    
    H: csr_matrix
        The partial parity check matrix for 1 type of stabilizers
        eg: Hz or Hx.


    
    Returns
    -------
    correction: np.ndarray
        An array with size N, where N is the total number of qubits in the surface code. This array has 1 at the index
        of each qubit that is to be corrected and 0 otherwise.

    """
    ## constructing the graphs and the syndromes of each cluster ##
    # These graphs are stored as an adjacency list that repeats the edges (since it stores the neighbours of each node), ie the
    # adjacency list format used: {0:[1,2], 1:[0,2], 2:[0,1]} ; 0,1,2 are the indices of the nodes.        
    # Note 1) if the edges are not repeated this will affect the edge removal in the tree finding function below (then need to add an
    # extra if function).
    # Note 2) in each graph the nodes (lattice vertices) are represented by their panqec integer indices.
    # Note 3) this step could be very expensive when it comes to the complexity 


    graphs = [] # list of all graphs organized per cluster in the form of a list of dictionaries
    sig = [] # list of all syndromes organized per cluster in the form of a list of lists
    syndromes_index = np.where(syndromes == 1)[0] # creating an array of indices of syndrome stabilizers

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
                if H[vertex,edge] == 1:
                    qubits_list.append(edge)

            # for each such qubit/edge, we find which other node from the cluster is attached to it. These
            # other nodes are the neighbours of the node "vertex"
            neighbours = []
            for qubit in qubits_list:
                for vertex_ in vertices_i: # we look at only the nodes already in the cluster discluding "vertex"--> np.where can be used                      
                    if vertex_ != vertex: # we don't want to consider the node "vertex" as its own neighbour
                        if H[vertex_, qubit] == 1:
                            neighbours.append(vertex_)                              
            
            graph_i[vertex] = neighbours

            ## organizing the syndromes per graph - for z syndromes

            if vertex in syndromes_index: # defiend externally
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
            n = nu_nodes(tree) # number of nodes of g
            nodes = list(tree.keys()) # list of nodes in the tree              
            s = nodes[0] # the first key (node) of the tree dictionary --> there could be a better way of finding the first key of "tree"!
            q = [] # empty list representing the que
            q.append(s) # adding s to the que 

            # creating "visited" and "parent" dictionaries which are both initialised as False for all the nodes
            values = [False]*n                 
            visited = {nodes[i]:values[i] for i in range (n)} # a dictionary to check if a node has been visited
            parent = copy.copy(visited) # a dictionary to check if a node is a parent node --> make sure to duplicate or create from scratch
            visited[s]= True # mark start node as being visited

            while len(q): # while the que is not empty

                current_node = q[0] # the node at the start of the que
                q.pop(0)  # remove "current_node" from the que
                parent[current_node] = True # "current_node" is now a parent node               
                
                neighbours_list = find_neighbours(tree,current_node)[:] # list of neighbours, duplicated so that it doesn't change as tree is modified
                for next in neighbours_list: # looping through the neighbours of "current_node" in the tree
                    if not parent[next]: # if a parent node we do nothing
                        if not visited[next]: # if node "next" was not visited before, add to the que                            
                            q.append(next)
                            visited[next] = True                        
                        else: # if node is visited
                            # remove the edge between "next" and "current_node" entirely from the tree graph
                            tree[current_node].remove(next) #--> maybe find another way but it is only upto 4 so ok
                            tree[next].remove(current_node) # --> find another way but only 4 so ok
            return tree
    ## Finding the forest:
    forest = [] # initialise the forest list
    index = 0 # variable to track printed output (to check which graph we are finding the tree for)
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

        while len(tree.keys())>1: # while tree dictionary is not empty. tree = {0:[]} is considered empty for us!!
            u = leaves[0] # choose pendant vertex to work with randomly (first element of "leaves")
            v = tree[u][0] # neighbour of the pendant vertix, which is the vertix connecting the leaf edge to the forest

            # removing leaf edge from "leaves" and from the tree.
            leaves.pop(0) # remove u from leaves --> u is leaf node 0 so we can use pop instead of remove
            tree.pop(u) # remove v from the tree --> not good for complexity? for dictionary it is of order of one access time so ok to lookup an item and remove
            tree[v].remove(u) # remove u from the neighbours list of v --> upto 4 elements in the list
            
            # checking if v is a pendant vertex now, and if so add to leaves
            if len(tree[v])==1:
                leaves.append(v)            

            # peeler algorithm            
            if u in sig_i:
                e = (u,v) # edge "e" to be corrected stored as a tuple e = (v1,v2) where v1 and v2 are the vertices bounding "e"
                A_.append(e)  
                sig_i.remove(u) 

                # flip v in sig_i
                if v in sig_i:
                    sig_i.remove(v)
                else:
                    sig_i.append(v)

    ## Now proccess A_ to find correction of the bsf ##

    ## Processing A_ to extract the edge/qubit indices in panqec --> better ways: H * H^T  or use panqec coordinates ( the average of the coordinates of v1 & v2, this can be a function)
    # note that is could be expensive when it comes to the complexity
    
    A = [] # list of indices for the qubits to be corrected
    # for each edge in A_, we loop through all the edges in H to see which edge is attached to the two vertices v1 and v2
    # in the tuple e = (v1,v2). Attached means we have a 1 in the H matrix.   
    N =  H[0].shape[1] # total number of qubits in the code
    for edge in A_:
        v1 = edge[0]
        v2 = edge[1]        
        for qubit in range(N): # --> 
            if (H[v1,qubit]==1) and (H[v2,qubit]==1): # if "qubit" attached to both v1 and v2
                A.append(qubit)

    ## Finding correction
    correction= np.zeros(N) # np.zeros(N)
    for qubit in A: # we find Ex by having 1 for the qubits that are to be corrected and 0 for the other qubits
        correction[qubit] = 1

    return correction
    