# coding:utf-8

#%% read in data - use a pandas dataframe just for convenience
import time
import pandas as pd

city = "smallLondon"
d = 36545
SIZE = 36545  # 3619
data = pd.read_table("data/" + city + "Graph.txt",
                     sep=" ",
                     header=None,
                     names=['vx', 'vy', 'weight'])



# centrality heuristic
import networkx as nx
graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)

# d : landmark number
# degreee heuristic
# degree = data.vx.value_counts()
# print(type(degree))
# landmarks = degree[0:d].index

import numpy as np
L = int(SIZE / 2)

# distanceMatrix = np.zeros((L, d), dtype=np.float16)
# for i in range(L):
#     if i in graph_dict.keys():
#         length = nx.single_source_dijkstra_path_length(G, i)
#         temp = []
#         for j in range(d):
#             if j in length.keys():
#                 distanceMatrix[i, j] = length[j]
#     print(str(i) + "th completed")
#     # print(str(i)," ",str(j)," ",length[j])
#     # distanceMatrix.dump(city+"LandmarkDistanceMatrix.dat")
# distanceMatrix.dump(city + "DistanceMatrix0.dat")


distanceMatrix = np.zeros((SIZE - L, d), dtype=np.float16)

for i in range(L, SIZE):
    if i in graph_dict.keys():
        length = nx.single_source_dijkstra_path_length(G, i)
        temp = []
        for j in range(d):
            if j in length.keys():
                distanceMatrix[i - L, j] = length[j]
    print(str(i) + "th completed")
    # print(str(i)," ",str(j)," ",length[j])
    # distanceMatrix.dump(city+"LandmarkDistanceMatrix.dat")
distanceMatrix.dump(city + "DistanceMatrix1.dat")
