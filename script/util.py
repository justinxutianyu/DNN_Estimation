import numpy as np
import pandas as pd
import networkx as nx
import city
########################  loading data and graph #######################
def load_data(City):
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                        sep=" ",
                        header=None,
                        names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
    # graph_nodes = graph.nodes()
    graph_dict = nx.to_dict_of_dicts(graph)
    G = nx.Graph(graph_dict)
    test_distance_matrix = np.load(City.location+"DistanceMatrix.dat")
    distance_matrix = np.load(City.location + "LandmarkDistanceMatrix.dat")
    print("Matrix is loaded")

    return (distance_matrix ,test_distance_matrix)

def shuffle(size):
    # random shuffle input data
    index = np.zeros(shape=(size * size, 2), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            index[i * size + j, 0] = i
            index[i * size + j, 1] = j
    np.random.shuffle(index)

    return index