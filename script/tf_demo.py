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

######################## prepare the data ########################
# X, y = load_linnerud(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

######################## time stamp ##################
timestr = time.strftime("%Y%m%d-%H%M%S")

######################## set learning variables ##################

SIZE =  8105 #3619 #  8105
test_Size = 8105 # 8105
learning_rate = 0.01
d =  8105 # 500
epochs = 1
Units = 1
batch_size = SIZE
location = "NewYork"
filename = location+"_NN"+timestr+"_"+str(Units)+"Units"+str(epochs)+"Epochs"+str(learning_rate)+"Rate"
########################  load training data #######################
edges = pd.read_table("data/"+location+"Graph.txt",
                    sep = " ",
                    header = None,
                    names = ['vx', 'vy', 'weight'])

graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)
# test_distanceMatrix = np.load(location+"DistanceMatrix.dat")
distanceMatrix = np.load(location+"DistanceMatrix.dat")
print("Matrix is loaded")

## d : landmark number
# degreee heuristic
degree = edges.vx.value_counts()
print(type(degree))
landmarks = degree[0:d].index

A = nx.to_numpy_matrix(G)
## random shuffle input data
# index = np.zeros(shape=(SIZE*SIZE, 2),dtype=np.int8)
# for i in range(SIZE):
#     for j in range(SIZE):
#         index[i*SIZE+j,0] = i
#         index[i*SIZE+j,1] = j
# np.random.shuffle(index)

######################## set some variables #######################
x = tf.placeholder(tf.float32, [None, 2*d], name='x')  # 3 features
y_ = tf.placeholder(tf.float32, [None, 1], name='y')  # 3 outputs

# hidden layer 1
# W1 = tf.Variable(tf.truncated_normal([2*d, 1], stddev=0.03), name='W1')
# b1 = tf.Variable(tf.truncated_normal([1]), name='b1')
W1 = tf.Variable(tf.truncated_normal([2*d, Units], mean=0.0, stddev=1), name='W1')
b1 = tf.Variable(tf.truncated_normal([1]), name='b1')

# hidden layer 2
# W2 = tf.Variable(tf.truncated_normal([10, 3], stddev=0.03), name='W2')
# b2 = tf.Variable(tf.truncated_normal([3]), name='b2')
W2 = tf.Variable(tf.truncated_normal([Units,1], mean=0.0, stddev=1), name='W2')
b2 = tf.Variable(tf.truncated_normal([1]), name='b2')


######################## Activations, outputs ######################
# output hidden layer 1
hidden_out = tf.nn.relu(tf.matmul(x, W1) + b1)

# total output
y = tf.nn.relu(tf.matmul(hidden_out, W2) + b2)

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)
error = tf.reduce_mean(tf.abs(tf.subtract(y,y_)))

####################### Optimizer      #########################
_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(_vars)

sys.exit(0)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

####################### Saver         #########################
saver = tf.train.Saver()

###################### Initialize, Accuracy and Run #################
# initialize variables
init_op = tf.global_variables_initializer()

# accuracy for the test set
accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_)))  # or could use tf.losses.mean_squared_error

# run
# with tf.Session() as sess:
#   sess.run(init_op)
#   total_batch = int(len(y_train) / batch_size)
#   for epoch in range(epochs):
#     avg_cost = 0
#     for i in range(total_batch):
#       batch_x, batch_y = X_train[i * batch_size:min(i * batch_size + batch_size, len(X_train)), :], \
#                          y_train[i * batch_size:min(i * batch_size + batch_size, len(y_train)), :]
#       _, c = sess.run([optimizer, mse], feed_dict={x: batch_x, y: batch_y})
#       avg_cost += c / total_batch
#     if epoch % 10 == 0:
#       print 'Epoch:', (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost)
#   print sess.run(mse, feed_dict={x: X_test, y: y_test})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, "data/"+location+"_nn_model"+str(learning_rate)+".ckpt")
    # total_batch = int(len(y_train) / batch_size)
    loss_array = []
    for k in range(epochs):
        index = np.zeros(shape=(SIZE*SIZE, 2),dtype=np.int8)
        for i in range(SIZE):
            for j in range(SIZE):
                index[i*SIZE+j,0] = i
                index[i*SIZE+j,1] = j
        np.random.shuffle(index)
        for i in range(SIZE):
            # batch_xs, batch_ys = # mnist.train.next_batch(100)
            print(str(i)+"th training")
            avg_cost = 0
            # vi = np.zeros(shape=(d))
            # for k in range(d):
            #     vi[k] = A[i,landmarks[k]]
            batch_xs = np.zeros(shape=(SIZE, 2*d))
            batch_ys = np.zeros(shape=(SIZE,1))
            for j in range(SIZE):
                vi = np.squeeze(np.asarray(distanceMatrix[index[i*SIZE+j ,0],:]))
                vj = np.squeeze(np.asarray(distanceMatrix[index[i*SIZE+j ,1],:]))
                # vi = np.zeros(shape=(d))
                # vj = np.zeros(shape=(d))
                # for k in range(d):
                #     vi[k] = distanceMatrix[index[i*SIZE+j ,0],landmarks[k]]
                #     vj[k] = distanceMatrix[index[i*SIZE+j ,0],landmarks[k]]
                # batch_xs = np.zeros(shape=(SIZE, 2*d))
                # batch_ys = np.zeros(shape=(SIZE))
                batch_xs[j] = np.concatenate([vi,vj])
                batch_ys[j] = distanceMatrix[index[i*SIZE+j ,0],index[i*SIZE+j ,1]]
            # print(batch_xs)
            # print(batch_ys)
            # print(sess.run(y, feed_dict={x: batch_xs}))
            _, c = sess.run([optimizer,mse], feed_dict={x: batch_xs, y_: batch_ys})
            # cost = (sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
            avg_cost = c/SIZE
            loss_array.append(avg_cost)
            #writer.add_summary(cost, i)
            print('train_step:', (i + 1), 'cost =', '{:.3f}'.format(avg_cost))

    plt.plot(loss_array)
    plt.ylabel('nn_loss')
    plt.savefig(filename+'_loss.png')
    plt.show()

    # Load testing data
    dif = []
    cost = 0
    mean_error = 0.0
    mean_error2 = 0.0
    true_distance = 0.0
    pred_distance = 0.0
    for i in range(test_Size):
        test_x = np.zeros(shape=(test_Size, 2*d))
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
            #     vi[k] = distanceMatrix[i,landmarks[k]]
            #     vj[k] = distanceMatrix[j,landmarks[k]]
            vi = np.squeeze(np.asarray(distanceMatrix[i, :]))
            vj = np.squeeze(np.asarray(distanceMatrix[j, :]))
            # vj = A[j,:]#np.zeros(shape=(d))
            # for m in range(d):
            #     vj[m] = A[j,landmarks[m]]
            # test_x[j] = np.hstack((vi,vj))
            test_x[j] = np.concatenate([vi,vj])
            test_y[j] = distanceMatrix[i,j] # use the origin matrix
            testx = np.reshape(test_x[j],(1, 2*d))
            testy = np.reshape(test_y[j],(1, 1))
            # e = sess.run(error, feed_dict={x: testx ,y_: testy})
            # mean_error = mean_error + e/(test_y[j] + 1)
            pred = sess.run(y, feed_dict={x: testx})
            pred = pred[0][0]
            preds.append(pred)
            y_true =  test_y[j]
            pred_y[j] = pred
            actual_y[j] = y_true            
            
            pred_distance += pred
            true_distance += y_true
            temp_error += abs(pred - y_true)/(y_true + 1)
            mean_error2 += abs(pred - y_true)/(y_true + 1)

            # error = tf.abs(tf.subtract(y, y_))
        print(preds)
        c = sess.run(mse, feed_dict={x: test_x,y_: test_y})
        avg_cost = c/SIZE
        dif.append(avg_cost)
        
        # mean_error = mean_error/test_Size
        # mean_error2 += temp_error
        print('test_step:', (i + 1), 'cost =', '{:.3f}'.format(avg_cost))
        temp_error = temp_error/test_Size
        print('test_step:', (i + 1), 'relative error =', temp_error)
        temp_error2 = mean_absolute_error(pred_y, actual_y)
        print('test_step:', (i + 1), 'abslute error =', temp_error2)
        mean_error += temp_error2
        # accuracy = tf.reduce_mean(tf.abs(tf.subtract(y, y_)))
        cost += avg_cost

        # print(sess.run(
        #     accuracy, feed_dict={
        #         x: test_x,
        #         y_: test_y
        #     }))
    print("MSE: ",cost/test_Size)
    print("Mean actual distance: ", true_distance/(test_Size*test_Size))
    print("Mean predicted distance: ", pred_distance/(test_Size*test_Size))
    print("Max average error: ", max(dif))
    print("Min average error: ", min(dif))
    print("Mean Absolute error", mean_error/test_Size)
    print("Mean relative error", mean_error2*100/(test_Size*test_Size))

    plt.scatter(range(len(dif)), dif, label='prediction')

    plt.xlabel('batch')
    plt.ylabel('error')
    plt.legend()

    plt.savefig(filename+"_test.png")
    plt.show()
    # Test trained model
    #correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
