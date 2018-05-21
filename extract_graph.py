# -*- coding: utf-8 -*-

import numpy as np


def extract_graph(city, nodes):
    #matrix = [[0 for x in range(nodes)] for y in range(nodes)]
    # matrix = np.zeros((edges, 3))
    matrix = np.zeros((nodes, nodes))
    print(matrix.shape)
    with open("../data/small.pypgr", "r") as f:
        for line in f:
            x = line.split(" ")
            if len(x) >= 6:
                matrix[int(x[0])][int(x[1])] = float(x[2])
                print(line)

    file = open("data/" + city + "Graph.txt", "w")
    for x in range(nodes):
        for y in range(nodes):
            if matrix[x][y] > 0:
                file.write(str(x) + " " + str(y) + " " +
                           str(matrix[x][y]) + "\n")
        print(str(x) + "th completed")
    file.close()


def extract_largeGraph(city, edges):
    matrix = np.zeros((edges, 3))
    print(matrix.shape)
    with open("../data/" + city + ".pypgr", "r") as f:
        index = 0
        for line in f:
            x = line.split(" ")
            if len(x) >= 6:
                matrix[index][0] = int(x[0])
                matrix[index][1] = int(x[1])
                matrix[index][2] = float(x[2])
                index += 1
                # print(line)
    # sort matrix in ascending error
    new_matrix = matrix[matrix[:, 0].argsort()]
    file = open("data/" + city + "Graph.txt", "w")
    # for x in range(nodes):
    #     for y in range(nodes):
    #         if matrix[x][y] > 0:
    #             file.write(str(x) + " " + str(y) + " " +
    #                        str(matrix[x][y]) + "\n")
    for i in range(new_matrix.shape[0]):
        file.write(str(int(new_matrix[i][0])) + " " +
                   str(int(new_matrix[i][1])) + " " + str(new_matrix[i][2]) + "\n")
        print(str(i) + "th completed")
    file.close()

edges = 2196
city = "small"
extract_largeGraph(city, edges)
