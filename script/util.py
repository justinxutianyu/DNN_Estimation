import numpy as np
import pandas as pd
import networkx as nx
import city
import os


def load_data(City, path):

    ########################  loading data and graph #######################
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                          sep=" ",
                          header=None,
                          names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
    # graph_nodes = graph.nodes()
    # graph_dict = nx.to_dict_of_dicts(graph)
    # G = nx.Graph(graph_dict)
    test_distance_matrix = np.load(os.path.join(
        path, City.location + "DistanceMatrix.dat"))
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


def shuffle_batch(size, batch):
    # random shuffle input data
    a = np.asarray(sorted(range(size) * batch), dtype=np.int32)
    b = np.asarray(range(batch) * size, dtype=np.int32)
    a = a.reshape(size * batch, 1)
    b = b.reshape(size * batch, 1)
    index = np.concatenate((a, b), axis=1)
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


def load_SL_data(City, path):

    ########################  loading data and graph #######################
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                          sep=" ",
                          header=None,
                          names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')

    temp1 = np.load(os.path.join(
        path, City.location + "DistanceMatrix0.dat"))
    temp2 = np.load(os.path.join(
        path, City.location + "DistanceMatrix1.dat"))
    test_distance_matrix = np.concatenate((temp1, temp2), axis=0)
    distance_matrix = np.load(os.path.join(
        path, City.location + "LandmarkDistanceMatrix.dat"))
    print("Matrix is loaded")

    ######################## preprocessing data #######################
    max_distance = np.amax(test_distance_matrix)
    distance_matrix = distance_matrix / max_distance
    test_distance_matrix = test_distance_matrix / max_distance

    return (distance_matrix, test_distance_matrix, max_distance)


def load_SL_adj(City, path):

    ########################  loading data and graph #######################
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                          sep=" ",
                          header=None,
                          names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')

    temp1 = np.load(os.path.join(
        path, City.location + "DistanceMatrix0.dat"))
    temp2 = np.load(os.path.join(
        path, City.location + "DistanceMatrix1.dat"))
    test_distance_matrix = np.concatenate((temp1, temp2), axis=0)
    adj_matrix = nx.to_numpy_matrix(graph)
    print("Matrix is loaded")

    ######################## preprocessing data #######################
    max_distance = np.amax(test_distance_matrix)
    max_edge = np.amax(adj_matrix)
    adj_matrix = adj_matrix / max_edge
    test_distance_matrix = test_distance_matrix / max_distance

    return (adj_matrix, test_distance_matrix, max_distance)


def load_SL_train(City, path):

    ########################  loading data and graph #######################
    edges = pd.read_table("data/" + City.location + "Graph.txt",
                          sep=" ",
                          header=None,
                          names=['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')

    temp1 = np.load(os.path.join(
        path, City.location + "DistanceMatrix0.dat"))
    temp2 = np.load(os.path.join(
        path, City.location + "DistanceMatrix1.dat"))
    test_distance_matrix = np.concatenate((temp1, temp2), axis=0)

    max_distance = np.amax(test_distance_matrix)
    test_distance_matrix = test_distance_matrix[:, :5000]

    distance_matrix = np.load(os.path.join(
        path, City.location + "LandmarkDistanceMatrix.dat"))
    print("Matrix is loaded")

    ######################## preprocessing data #######################
    max_distance = np.amax(test_distance_matrix)
    distance_matrix = distance_matrix / max_distance
    test_distance_matrix = test_distance_matrix / max_distance

    return (distance_matrix, test_distance_matrix, max_distance)
