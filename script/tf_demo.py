# Trying to define the simplest possible neural net where the output layer of the neural net is a single
# neuron with a "continuous" (a.k.a floating point) output.  I want the neural net to output a continuous
# value based off one or more continuous inputs.  My real problem is more complex, but this is the simplest
# representation of it for explaining my issue.  Even though I've oversimplified this to look like a simple
# linear regression problem (y=m*x), I want to apply this to more complex neural nets.  But if I can't get
# it working with this simple problem, then I won't get it working for anything more complex.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

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

######################## set learning variables ##################
SIZE = 3619 # 8105 
test_Size = 3619 # 8105
learning_rate = 0.01
d =  3619 # 8105 # 500
epochs = SIZE
batch_size = SIZE
location = "Melbourne"
filename = location+"nn_allDistance_"+str(learning_rate)
########################  load training data #######################
edges = pd.read_table("data/"+location+"Graph.txt",
                    sep = " ",
                    header = None,
                    names = ['vx', 'vy', 'weight'])

graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)
distanceMatrix = np.load(location+"DistanceMatrix.dat")
print("Matrix is loaded")

######################## set some variables #######################
x = tf.placeholder(tf.float32, [None, 2*d], name='x')  # 3 features
y_ = tf.placeholder(tf.float32, [None, 1], name='y')  # 3 outputs

# hidden layer 1
# W1 = tf.Variable(tf.truncated_normal([2*d, 1], stddev=0.03), name='W1')
# b1 = tf.Variable(tf.truncated_normal([1]), name='b1')
W1 = tf.Variable(tf.truncated_normal([2*d, 1]), name='W1')
b1 = tf.Variable(tf.truncated_normal([1]), name='b1')

# hidden layer 2
# W2 = tf.Variable(tf.truncated_normal([10, 3], stddev=0.03), name='W2')
# b2 = tf.Variable(tf.truncated_normal([3]), name='b2')
W2 = tf.Variable(tf.truncated_normal([1,1]), name='W2')
b2 = tf.Variable(tf.truncated_normal([1]), name='b2')


######################## Activations, outputs ######################
# output hidden layer 1
hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

# total output
y = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)
error = tf.reduce_mean(tf.abs(tf.subtract(y,y_)))

####################### Optimizer      #########################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

####################### Saver         #########################
saver = tf.train.Saver()

###################### Initialize, Accuracy and Run #################
# initialize variables
init_op = tf.global_variables_initializer()

# accuracy for the test set
accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_)))  # or could use tf.losses.mean_squared_error

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, "data/"+location+"_nn_model"+str(learning_rate)+".ckpt")
    # total_batch = int(len(y_train) / batch_size)
    loss_array = []
    # Load testing data
    dif = []
    cost = 0
    mean_error = 0.0
    for i in range(test_Size):
        test_x = np.zeros(shape=(test_Size, 2*d))
        test_y = np.zeros(shape=(test_Size, 1))
        # vi = np.zeros(shape=(d))
        # for k in range(d):
        #     vi[k] = A[i,landmarks[k]]
        # vi = A[i,:]
        avg_cost = 0
        for j in range(test_Size):
            # vi = np.zeros(shape=(d))
            # vj = np.zeros(shape=(d))
            # for k in range(d):
            #     vi[k] = distanceMatrix[i,landmarks[k]]
            #     vj[k] = distanceMatrix[j,landmarks[k]]
            vi = np.squeeze(np.asarray(distanceMatrix[i, :]))
            vj = np.squeeze(np.asarray(distanceMatrix[j, :]))
            # vj = A[j,:]#np.zeros(shape=(d))
            # for m in range(d):
            #     vj[m] = A[j,landmarks[m]]
            # test_x[j] = np.hstack((vi,vj))
            test_x[j] = np.concatenate([vi,vj])
            test_y[j] = distanceMatrix[i,j]
            testx = np.reshape(test_x[j],(1, 2*d))
            testy = np.reshape(test_y[j],(1, 1))
            # print(x.shape)
            # print(y)
            e = sess.run(error, feed_dict={x: testx ,y_: testy})
            mean_error = mean_error + e/test_y[j]
            print(e)