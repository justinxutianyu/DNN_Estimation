# coding:utf-8
import time
import pandas as pd
import numpy as np
import city
import networkx as nx
import os


def load_graph(City):
    data = pd.read_table("data/" + City.location + "Graph.txt",
                         sep=" ",
                         header=None,
                         names=['vx', 'vy', 'weight'])
    return data


def get_landmarks(data, d):
    # d : landmark number
    # degreee heuristic
    degree = data.vx.value_counts()
    print(type(degree))
    landmarks = degree[0:d].index

    return landmarks


def distribute_distance(City, data):
    size = City.size
    d = City.d
    n = 2
    block = int(size / n)
    graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
    graph_dict = nx.to_dict_of_dicts(graph)
    G = nx.Graph(graph_dict)

    distanceMatrix = np.zeros((block, d), dtype=np.float16)
    for i in range(block):
        if i in graph_dict.keys():
            length = nx.single_source_dijkstra_path_length(G, i)
            temp = []
            for j in range(d):
                if j in length.keys():
                    distanceMatrix[i, j] = length[j]
        print(str(i) + "th completed")
    distanceMatrix.dump(city + "DistanceMatrix0.dat")

    distanceMatrix = np.zeros((size - block, d), dtype=np.float16)
    for i in range(block, size):
        if i in graph_dict.keys():
            length = nx.single_source_dijkstra_path_length(G, i)
            temp = []
            for j in range(d):
                if j in length.keys():
                    distanceMatrix[i - block, j] = length[j]
        print(str(i) + "th completed")
        # print(str(i)," ",str(j)," ",length[j])
        # distanceMatrix.dump(city+"LandmarkDistanceMatrix.dat")
    distanceMatrix.dump(city + "DistanceMatrix1.dat")


def generate_landmark_matrix(City, data, path):
    size = City.size
    d = City.d
    location = city.location

    # generate graph
    graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
    graph_dict = nx.to_dict_of_dicts(graph)
    G = nx.Graph(graph_dict)

    # generate landmarks
    landmarks = get_landmarks(data, d)

    distanceMatrix = np.zeros((size, d), dtype=np.float16)
    for i in range(size):
        if i in graph_dict.keys():
            length = nx.single_source_dijkstra_path_length(G, i)
            temp = []
            for j in range(d):
                if landmarks[j] in length.keys():
                    distanceMatrix[i, j] = length[landmarks[j]]
        print(str(i) + "th completed")
    distanceMatrix.dump(os.path.join(path,
                                     location + "LandmarkDistanceMatrix.dat"))

city = city.City('SL')
data = load_graph(city)
path = "/mnt/Project/data"
generate_landmark_matrix(city, data, path)
