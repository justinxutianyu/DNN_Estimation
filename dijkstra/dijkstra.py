# -*- coding: utf-8 -*-

# %% define the dijkstra function
def dijkstra(graph_dict, start, end):
    """
    This is a recursive function that implements Dijkstra's Shortest Path
    algorithm.
    
    It takes as its inputs:
        i. a graph represented by a "dictionary of dictionaries" structure,
            generated using networkx;
        ii. a starting node in that graph; and
        iii. an ending node for that graph
        
    It then performs the following steps:
            i. initialises a set of distances from the start node as infinity;
            ii. initialises a set of 'predecessors' to None (a predecessor is
                defined for each node in the network and it lists the prior node
                in the path from start to end);
            iii. initialises the set of of vertices for which the shortest path
            from start to end has been found to empty; and then
            iv. whilst there are still vertices left to assess:
                a. restricts the set of vertices to those where that still need
                    analysisng;
                b. finds the vertex that is the minimum distance from the start;
                c. "relaxes" the neighbours of this closest vertex to see if the
                    shortest path to that vertex can be improved; and
                d. updates the predecessor vertex for each node in the current path
            
            When all vertices have been assessed, the function defines the path
            and returns it with its associated cost
    """
    distances = {} # empty dict for distances
    predecessors = {} # list of vertices in path to current vertex
    
    to_assess = graph_dict.keys() # get all the nodes in the graph that need to be assessed

    # set all initial distances to infinity and no predecessor for any node
    for node in graph_dict:
        distances[node] = float('inf')
        predecessors[node] = None
    
    # set the intial collection of permanently labelled nodes to be empty
    sp_set = []
    # set the distance from the start node to be 0
    distances[start] = 0
    
    # as long as there are still nodes to assess:
    while len(sp_set) < len(to_assess):
        # chop out any nodes with a permament label
        still_in = { node: distances[node] for node in [node for node in to_assess if node not in sp_set] }
        # find the closest node to the current node
        closest = min(still_in, key = distances.get)
        # and add it to the set of permanently labelled nodes
        sp_set.append(closest)
        
        # then for all the neighbours of the closest node (that was just added)
        # to the permanent set
        for node in graph_dict[closest]:
            # if a shorter path to that node can be found
            if distances[node] > distances[closest] + graph[closest][node]['weight']:
                # update the distance with that shorter distance; and
                distances[node] = distances[closest] + graph[closest][node]['weight']
                # set the predecessor for that node
                predecessors[node] = closest
                
    # once the loop is complete the final path needs to be calculated - this can
    # be done by backtracing through the predecessors set
    path = [end]
    while start not in path:
        path.append(predecessors[path[-1]])
    
    # return the path in order start -> end, and it's cost
    #return path[::-1], distances[end]
    return distances[end]

# %%
# function to get _all_ dijkstra shortest paths
def dijkstra_all(graph_dict):
    ans = []
    for start in graph_dict.keys():
        for end in graph_dict.keys():
            ans.append(dijkstra(graph_dict, start, end))
    return ans            


       
#%% read in data - use a pandas dataframe just for convenience
import pandas as pd
data = pd.read_table("~/Desktop/Project/data/melbourne_graph.txt",
                     sep = " ",
                     header = None, 
                     names = ['vx', 'vy', 'weight'])

# %% use network x to prepare dictionary structure which can be fed in to the 
# dijkstra function
import networkx as nx
graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)
# %% run the functions
print graph_dict
#all_paths = dijkstra_all(graph_dict)
result = []
#length = dict(nx.all_pairs_dijkstra_path_length(G))
length = nx.single_source_dijkstra_path_length(G, 0)
print "start compute length"
with open('distance.txt','w') as f:
    for i in graph_dict.keys():
        length = nx.single_source_dijkstra_path_length(G, i)
        for j in graph_dict.keys():
            f.write(str(i)+" "+str(j)+" "+str(length[j]))
                
