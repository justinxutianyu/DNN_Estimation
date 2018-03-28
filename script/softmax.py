from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx

import logging
logging.getLogger().setLevel(logging.INFO)

FLAGS = None
SIZE = 3619
test_Size = 3619#13129

def main(_):
    # Import data
    #   mnist = input_data.read_data_sets(FLAGS.data_dir)

    # load training data
    edges = pd.read_table("data/demoGraph.txt",
                        sep = " ",
                        header = None,
                        names = ['vx', 'vy', 'weight'])

    graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
    # graph_nodes = graph.nodes()
    graph_dict = nx.to_dict_of_dicts(graph)
    G = nx.Graph(graph_dict)
    distanceMatrix = np.load("demoDistanceMatrix.dat")
    print("Matrix is loaded")

    # d : landmark number
    d = SIZE
    # degreee heuristic
    degree = edges.vx.value_counts()
    print(type(degree))
    landmarks = degree[0:d].index

    # Create the model
    x = tf.placeholder(tf.float32, [None, 2*d])
    W = tf.Variable(tf.zeros([2*d, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # outputs of 'y', and then average across the batch.
    # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # Same training as used in mnist example
    loss = tf.reduce_mean(tf.square(y_ - y))
    # cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    sess = tf.InteractiveSession()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()
    saver.save(sess, "data/model.ckpt")

    # Train
    A = nx.to_numpy_matrix(G)
    for i in range(SIZE):
        # batch_xs, batch_ys = # mnist.train.next_batch(100)
        print(str(i)+"th training")
        vi = np.squeeze(np.asarray(A[i,:]))
        batch_xs = np.zeros(shape=(SIZE, 2*d))
        batch_ys = np.zeros(shape=(SIZE))
        # vi = np.zeros(shape=(d))
        # for k in range(d):
        #     vi[k] = A[i,landmarks[k]]
        for j in range(SIZE):
            vj = np.squeeze(np.asarray(A[j,:]))
            batch_xs[j] = np.concatenate([vi,vj])
            # vj = np.zeros(shape=(d))
            # for m in range(d):
            #     vj[m] = A[j,landmarks[m]]
            # batch_xs[j] = np.hstack((vi,vj))
            batch_ys[j] = distanceMatrix[i,j]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(sess.run(loss))

    # Load testing data
    test_edges = pd.read_table("data/demoGraph.txt",
                        sep = " ",
                        header = None,
                        names = ['vx', 'vy', 'weight'])

    test_graph = nx.from_pandas_edgelist(test_edges, 'vx', 'vy', 'weight')
    # graph_nodes = graph.nodes()
    test_graph_dict = nx.to_dict_of_dicts(test_graph)
    test_G = nx.Graph(test_graph_dict)
    test_distanceMatrix = np.load("demoDistanceMatrix.dat")
    A = nx.to_numpy_matrix(test_G)
    # batch_xs, batch_ys = # mnist.train.next_batch(100)
    #vi = np.squeeze(np.asarray(A[0,:]))

    for i in range(test_Size):
        test_x = np.zeros(shape=(test_Size, 2*d))
        test_y = np.zeros(shape=(test_Size))
        vi = np.squeeze(np.asarray(A[i,:]))
        # vi = np.zeros(shape=(d))
        # for k in range(d):
        #     vi[k] = A[i,landmarks[k]]
        # vi = A[i,:]
        for j in range(test_Size):
            vj = np.squeeze(np.asarray(A[j,:]))
            # vj = A[j,:]#np.zeros(shape=(d))
            # for m in range(d):
            #     vj[m] = A[j,landmarks[m]]
            # test_x[j] = np.hstack((vi,vj))
            test_x[j] = np.concatenate([vi,vj])
            test_y[j] = test_distanceMatrix[i,j]

        accuracy = tf.subtract(y, y_)
        print(sess.run(
            accuracy, feed_dict={
                x: test_x,
                y_: test_y
            }))
    # Test trained model
    #correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
