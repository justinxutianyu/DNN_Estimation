import numpy as np
import pandas as pd
import networkx as nx
import city


def load_data(City):

    ########################  loading data and graph #######################
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                          sep=" ",
                          header=None,
                          names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
    # graph_nodes = graph.nodes()
    # graph_dict = nx.to_dict_of_dicts(graph)
    # G = nx.Graph(graph_dict)
    test_distance_matrix = np.load(City.location + "DistanceMatrix.dat")
    distance_matrix = np.load(City.location + "LandmarkDistanceMatrix.dat")
    print("Matrix is loaded")

    ######################## preprocessing data #######################
    max_distance = np.amax(test_distance_matrix)
    distance_matrix = distance_matrix / max_distance
    test_distance_matrix = test_distance_matrix / max_distance

    return (distance_matrix, test_distance_matrix)


def shuffle(size):
    # random shuffle input data
    index = np.zeros(shape=(size * size, 2), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            index[i * size + j, 0] = i
            index[i * size + j, 1] = j
    np.random.shuffle(index)

    return index


def load_adj_data(City):
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                          sep=" ",
                          header=None,
                          names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
    # graph_nodes = graph.nodes()
    # G = nx.Graph(graph_dict)
    test_distance_matrix = np.load(City.location + "DistanceMatrix.dat")
    adj_matrix = nx.to_numpy_matrix(graph)
    print("Matrix is loaded")

    ######################## preprocessing data #######################
    max_distance = np.amax(test_distance_matrix)
    max_edge = np.amax(adj_matrix)
    adj_matrix = adj_matrix / max_edge
    # change no-connect edge to 1
    np.place(adj_matrix, adj_matrix == 0.0, 1.0)
    test_distance_matrix = test_distance_matrix / max_distance

    return (adj_matrix, test_distance_matrix, max_distance)
