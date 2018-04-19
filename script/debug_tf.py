######################### import stuff ##########################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
# import matplotlib.pyplot as plt
import matplotlib
import logging
logging.getLogger().setLevel(logging.INFO)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import check_array
from sklearn.metrics import mean_absolute_error

######################## time stamp ##################
timestr = time.strftime("%Y%m%d-%H%M%S")

######################## set parameters ##################

SIZE = 3619  # 3619 
test_Size = 3619 
learning_rate = 0.001
d = 500  # 500
epochs = 20
Units = 100
batch_size = SIZE
location = "Melbourne"

# SIZE = 8105
# test_Size = 8105
# learning_rate = 0.001
# d = 8105  # 500
# epochs = 1
# Units = 100
# batch_size = SIZE
# location = "NewYork"

filename = location + "_NN" + timestr + "_" + \
    str(Units) + "Units" + str(epochs) + "Epochs" + str(learning_rate) + "Rate"

########################  loading data and graph #######################
edges = pd.read_table("data/" + location + "Graph.txt",
                      sep=" ",
                      header=None,
                      names=['vx', 'vy', 'weight'])

graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)
test_distance_matrix = np.load(location+"DistanceMatrix.dat")
distance_matrix = np.load(location + "LandmarkDistanceMatrix.dat")
print("Matrix is loaded")

######################## preprocessing data #######################
max_distance = np.amax(test_distance_matrix)
distance_matrix = distance_matrix / max_distance
test_distance_matrix = test_distance_matrix / max_distance
# # d : landmark number
# # degreee heuristic
# degree = edges.vx.value_counts()
# print(type(degree))
# landmarks = degree[0:d].index
# A = nx.to_numpy_matrix(G)

# random shuffle input data
index = np.zeros(shape=(SIZE * SIZE, 2), dtype=np.int8)
for i in range(SIZE):
    for j in range(SIZE):
        index[i * SIZE + j, 0] = i
        index[i * SIZE + j, 1] = j
np.random.shuffle(index)

######################## set some variables #######################
x = tf.placeholder(tf.float32, [None, 2 * d], name='x')  # inpute features
y_ = tf.placeholder(tf.float32, [None, 1], name='y')  # predictions

# hidden layer 1
W1 = tf.Variable(tf.truncated_normal(
    [2 * d, Units], mean=0.0, stddev=0.01), name='W1')
b1 = tf.Variable(tf.truncated_normal([Units]), name='b1')

# hidden layer 2
W2 = tf.Variable(tf.truncated_normal(
    [Units, 1], mean=0.0, stddev=0.01), name='W2')
b2 = tf.Variable(tf.truncated_normal([1]), name='b2')


######################## Activations, outputs ######################
# output hidden layer 1
hidden_out = tf.nn.relu(tf.matmul(x, W1) + b1)

# total output
y = tf.nn.sigmoid(tf.matmul(hidden_out, W2) + b2)

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)
# error = tf.reduce_mean(tf.abs(tf.subtract(y,y_)))

####################### Optimizer      #########################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# gradients_op = tf.gradients(mse, trainable_vars)

# apply_gradients = optimizer.apply_gradients(zip(gradients_op, trainable_vars))
# print(trainable_vars)
# print("--" * 50)
# print(gradients_op)
# print("--" * 50)
# print(apply_gradients)
# print("--" * 50)

# sys.exit(0)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(mse)

# ####################### Saver         #########################
saver = tf.train.Saver()

# ###################### Initialize, Accuracy and Run #################
# # initialize variables
# init_op = tf.global_variables_initializer()
#
print('Start training')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, "data/" + location + "_nn_model" +
               str(learning_rate) + ".ckpt")
    # total_batch = int(len(y_train) / batch_size)
    loss_array = []
    for k in range(epochs):
        index = np.zeros(shape=(SIZE * SIZE, 2), dtype=np.int8)
        for i in range(SIZE):
            for j in range(SIZE):
                index[i * SIZE + j, 0] = i
                index[i * SIZE + j, 1] = j
        np.random.shuffle(index)
        for i in range(SIZE):
            # batch_xs, batch_ys = # mnist.train.next_batch(100)
            print(str(i) + "th training")
            avg_cost = 0
            # vi = np.zeros(shape=(d))
            # for k in range(d):
            #     vi[k] = A[i,landmarks[k]]
            batch_xs = np.zeros(shape=(SIZE, 2 * d))
            batch_ys = np.zeros(shape=(SIZE, 1))
            for j in range(SIZE):
                vi = np.squeeze(np.asarray(
                    distance_matrix[index[i * SIZE + j, 0], :]))
                vj = np.squeeze(np.asarray(
                    distance_matrix[index[i * SIZE + j, 1], :]))

                batch_xs[j] = np.concatenate([vi, vj])
                batch_ys[j] = test_distance_matrix[
                    index[i * SIZE + j, 0], index[i * SIZE + j, 1]]

            # print(sess.run(y, feed_dict={x: batch_xs}))
            _, c = sess.run([optimizer, mse], feed_dict={
                            x: batch_xs, y_: batch_ys})
            # cost = (sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
            # avg_cost = c/SIZE
            loss_array.append(c)
            #writer.add_summary(cost, i)
            print('train_step:', (i + 1), 'cost =', '{:.6f}'.format(c))

    plt.plot(loss_array)
    plt.ylabel('nn_loss')
    plt.savefig("picture/" + filename + '_loss.png')
    # plt.show()

    # Load testing data
    dif = []
    cost = 0
    mean_error = 0.0
    mean_error2 = 0.0
    true_distance = 0.0
    pred_distance = 0.0
    for i in range(test_Size):
        test_x = np.zeros(shape=(test_Size, 2 * d))
        test_y = np.zeros(shape=(test_Size, 1))
        # vi = np.zeros(shape=(d))
        # for k in range(d):
        #     vi[k] = A[i,landmarks[k]]
        # vi = A[i,:]
        avg_cost = 0
        pred_y = np.zeros(shape=(test_Size))
        actual_y = np.zeros(shape=(test_Size))
        temp_error = 0.0
        preds = []
        for j in range(test_Size):
            # vi = np.zeros(shape=(d))
            # vj = np.zeros(shape=(d))
            # for k in range(d):
            #     vi[k] = distance_matrix[i,landmarks[k]]
            #     vj[k] = distance_matrix[j,landmarks[k]]
            vi = np.squeeze(np.asarray(distance_matrix[i, :]))
            vj = np.squeeze(np.asarray(distance_matrix[j, :]))
            # vj = A[j,:]#np.zeros(shape=(d))
            # for m in range(d):
            #     vj[m] = A[j,landmarks[m]]
            # test_x[j] = np.hstack((vi,vj))
            test_x[j] = np.concatenate([vi, vj])
            test_y[j] = test_distance_matrix[i, j]  # use the origin matrix
            testx = np.reshape(test_x[j], (1, 2 * d))
            testy = np.reshape(test_y[j], (1, 1))
            # e = sess.run(error, feed_dict={x: testx ,y_: testy})
            # mean_error = mean_error + e/(test_y[j] + 1)
            pred = sess.run(y, feed_dict={x: testx})
            pred = pred[0][0]
            preds.append(pred)
            y_true = test_y[j, 0]
            pred_y[j] = pred * max_distance
            actual_y[j] = y_true * max_distance

            if y_true != 0:
                pred_distance += pred_y[j]
                true_distance += actual_y[j]
                temp_relative_error = abs(
                    pred_y[j] - actual_y[j]) / (actual_y[j] + 1)
                # print(temp_relative_error)

                # filter larger ratio
                temp_relative_error = min(temp_relative_error, 1.0)
                temp_error += temp_relative_error
                mean_error2 += temp_relative_error

            # error = tf.abs(tf.subtract(y, y_))
        # print(preds)
        c = sess.run(mse, feed_dict={x: test_x, y_: test_y})
        dif.append(c)

        # mean_error = mean_error/test_Size
        # mean_error2 += temp_error
        print('test_step:', (i + 1),
              'mean squared error =', '{:.6f}'.format(c))
        temp_error = temp_error / test_Size
        print('test_step:', (i + 1), 'relative error =', temp_error * 100)
        temp_error2 = mean_absolute_error(pred_y, actual_y)
        mean_error += temp_error2
        print('test_step:', (i + 1), 'abslute error =', temp_error2 )

        relative_error2 = np.abs(np.mean(pred_y) -
                                 np.mean(actual_y)) / np.mean(actual_y)
        print('test_step:', (i + 1), 'relative error2 =', relative_error2 * 100)

        # accuracy = tf.reduce_mean(tf.abs(tf.subtract(y, y_)))
        cost += c

    print("MSE: ", cost / test_Size)
    print("Max distance", max_distance)
    print("Mean actual distance: ", true_distance / (test_Size * test_Size))
    print("Mean predicted distance: ", pred_distance / (test_Size * test_Size))
    print("Max average error: ", max(dif))
    print("Min average error: ", min(dif))
    print("Mean Absolute error", mean_error / test_Size)
    print("Mean relative error", mean_error2 * 100 / (test_Size * test_Size))

    plt.plot(dif)

    plt.xlabel('batch')
    plt.ylabel('error')
    plt.legend()

    plt.savefig("picture/" + filename + "_test.png")
    # plt.show()
