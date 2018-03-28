#%% read in data - use a pandas dataframe just for convenience
import time
import pandas as pd

data = pd.read_table("~/Desktop/Project/data/melbourne_graph.txt",
                    sep = " ",
                    header = None, 
                    names = ['vx', 'vy', 'weight'])

# d : landmark number
d = 1000
# degreee heuristic
degree = data.vx.value_counts()
print(type(degree))
landmarks = degree[0:1000]

# centrality heuristic
import networkx as nx
graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)

start_time = time.time()
print "start compute centrality"
with open('../data/closeness.txt','a') as f:
    for i in range(34846,48513):
        if i in graph_dict.keys():
            length = nx.single_source_dijkstra_path_length(G, i)
            total_length = 0.0
            for j in graph_dict.keys():
                total_length += length[j]
            f.write(str(i)+" "+str(total_length)+"\n")
print(time.time()-start_time)