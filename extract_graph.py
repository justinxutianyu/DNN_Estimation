# -*- coding: utf-8 -*-

import numpy as np
nodes = 36545
#matrix = [[0 for x in range(nodes)] for y in range(nodes)]

matrix = np.zeros((nodes, nodes))
print(matrix.shape)
with open("data/bigMelbourne.pypgr", "r") as f:
    for line in f:
        x = line.split(" ")
        if len(x) >= 6:
            matrix[int(x[0])][int(x[1])] = float(x[2])
            print(line)

file = open("data/bigMelbourneGraph.txt", "w")
for x in range(nodes):
    for y in range(nodes):
        if matrix[x][y] > 0:
            file.write(str(x) + " " + str(y) + " " + str(matrix[x][y]) + "\n")
    print(str(x) + "th completed")
file.close()
