# coding:utf-8

#%% read in data - use a pandas dataframe just for convenience
import time
import pandas as pd

data = pd.read_table("data/MelbourneGraph.txt",
                    sep = " ",
                    header = None,
                    names = ['vx', 'vy', 'weight'])

# centrality heuristic
import networkx as nx
graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)

## d : landmark number
# degreee heuristic
d = 500
degree = graph.vx.value_counts()
print(type(degree))
landmarks = degree[0:d].index

import numpy as np
distanceMatrix = np.zeros((3619, 500))
for i in graph_dict.keys():
    length = nx.single_source_dijkstra_path_length(G, i)
    for j in range(d):
        distanceMatrix[int(i),int(landmarks[j])] = length[landmarks[j]]
    print(str(i)+"th completed")

distanceMatrix.dump("MelbourneLandmarkDistanceMatrix.dat")
