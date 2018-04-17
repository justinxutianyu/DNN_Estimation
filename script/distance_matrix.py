# coding:utf-8

#%% read in data - use a pandas dataframe just for convenience
import time
import pandas as pd

city = "London"
d = 89127
SIZE = 89127 # 3619
data = pd.read_table("data/"+city+"Graph.txt",
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
# degree = data.vx.value_counts()
# print(type(degree))
# landmarks = degree[0:d].index

import numpy as np
distanceMatrix = np.zeros((SIZE, d))
for i in graph_dict.keys():
    length = nx.single_source_dijkstra_path_length(G, i)
    for j in range(d):
        distanceMatrix[int(i),int(j)] = length[j]
        print(str(i)," ",str(j)," ",length[j])
    print(str(i)+"th completed")

# distanceMatrix.dump(city+"LandmarkDistanceMatrix.dat")
