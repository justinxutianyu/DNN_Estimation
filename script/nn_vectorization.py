######################### import stuff ##########################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import logging
import matplotlib
logging.getLogger().setLevel(logging.INFO)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import util
import city
######################## set parameters ##################
timestr = time.strftime("%Y%m%d-%H%M%S")

# Intialize class city
city = city.City('SL')

filename = city.name(timestr)

########################  loading data and graph #######################
path = "/mnt/Project/data"
distance_matrix, test_distance_matrix, max_distance = util.load_SL_train(
    city, path)


######################## shuffle input #######################
# index = util.shuffle(city.size)

######################## set some variables #######################
x = tf.placeholder(tf.float32, [None, 2 * city.d], name='x')  # inpute features
y_ = tf.placeholder(tf.float32, [None, 1], name='y')  # predictions

# hidden layer 1
W1 = tf.Variable(tf.truncated_normal(
    [2 * city.d, city.unit], mean=0.0, stddev=0.01), name='W1')
b1 = tf.Variable(tf.truncated_normal([city.unit]), name='b1')

# hidden layer 2
W2 = tf.Variable(tf.truncated_normal(
    [city.unit, 1], mean=0.0, stddev=0.01), name='W2')
b2 = tf.Variable(tf.truncated_normal([1]), name='b2')


######################## Activations, outputs ######################
# output hidden layer 1
hidden_out = tf.nn.relu(tf.matmul(x, W1) + b1)

# total output
y = tf.nn.sigmoid(tf.matmul(hidden_out, W2) + b2)

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)
# error = tf.reduce_mean(tf.abs(tf.subtract(y,y_)))

# sys.exit(0)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=city.learning_rate).minimize(mse)

# ####################### Saver         #########################
saver = tf.train.Saver()

# ###################### Initialize, Accuracy and Run #################
print('Start training')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, "model/" + filename + "_model" + ".ckpt")
    loss_array = []
    size = city.size
    d = city.d
    epoch = city.epoch
    batch = city.batch_size
    for k in range(epoch):
        index = util.shuffle_batch(size, batch)
        for i in range(size):
            # batch_xs, batch_ys = # mnist.train.next_batch(100)
            # print(str(i+1) + "th training")
            batch_xs = np.zeros(shape=(batch, 2 * d))
            batch_ys = np.zeros(shape=(batch, 1))
            for j in range(batch):
                vi = np.squeeze(np.asarray(
                    distance_matrix[index[i * batch + j, 0], :]))
                vj = np.squeeze(np.asarray(
                    distance_matrix[index[i * batch + j, 1], :]))

                batch_xs[j] = np.concatenate([vi, vj])
                batch_ys[j] = test_distance_matrix[
                    index[i * batch + j, 0], index[i * batch + j, 1]]

            _, c = sess.run([optimizer, mse], feed_dict={
                            x: batch_xs, y_: batch_ys})
            loss_array.append(c)
            print('train_step:', (i + 1), 'cost =', '{:.6f}'.format(c))

    plt.plot(loss_array)
    plt.ylabel('nn_loss')
    plt.savefig("picture/" + filename + '_loss.png')

    # Load testing data
    print('Start testing')
    distance_matrix, test_distance_matrix, max_distance = util.load_SL_test(
        city, path)

    test_size = city.test_size
    d = city.d
    batch = city.batch_size
    # temporal variable
    dif = []
    cost = 0
    absolute_error = 0.0
    relative_error = 0.0
    true_distance = 0.0
    pred_distance = 0.0
    relative_error_list = []
    absolute_error_list = []
    for i in range(test_size):
        test_x = np.zeros(shape=(batch, 2 * d))
        test_y = np.zeros(shape=(batch, 1))
        avg_cost = 0
        pred_y = np.zeros(shape=(batch))
        actual_y = np.zeros(shape=(batch))
        batch_relative_error = 0.0
        for j in range(batch):
            vi = np.squeeze(np.asarray(distance_matrix[i, :]))
            vj = np.squeeze(np.asarray(distance_matrix[j, :]))
            test_x[j] = np.concatenate([vi, vj])
            test_y[j] = test_distance_matrix[i, j]  # use the origin matrix
            testx = np.reshape(test_x[j], (1, 2 * d))
            testy = np.reshape(test_y[j], (1, 1))
            pred = sess.run(y, feed_dict={x: testx})[0, 0]
            # pred = pred[0][0]
            y_true = test_y[j, 0]
            pred_y[j] = pred * max_distance
            actual_y[j] = y_true * max_distance

            if y_true != 0:
                pred_distance += pred_y[j]
                true_distance += actual_y[j]
                temp = abs(
                    pred_y[j] - actual_y[j]) / (actual_y[j] + 1)
                # filter larger ratio
                temp = min(temp, 1.0)
                batch_relative_error += temp
                relative_error += temp

            # error = tf.abs(tf.subtract(y, y_))
        c = sess.run(mse, feed_dict={x: test_x, y_: test_y})
        dif.append(c)

        print('test_step:', (i + 1),
              'mean squared error =', '{:.6f}'.format(c))
        batch_relative_error = batch_relative_error / batch * 100
        relative_error_list.append(batch_relative_error)
        print('test_step:', (i + 1), 'relative error =',
              '{:.6f}'.format(batch_relative_error))
        e = mean_absolute_error(pred_y, actual_y)
        absolute_error += e
        absolute_error_list.append(e)
        print('test_step:', (i + 1), 'absolute error =', '{:.6f}'.format(e))

        cost += c

    print("MSE: ", cost / test_size)
    print("Max distance", max_distance)
    print("Mean actual distance: ", true_distance / (test_size * batch))
    print("Mean predicted distance: ", pred_distance / (test_size * batch))
    print("Max average error: ", max(dif))
    print("Min average error: ", min(dif))
    print("Mean Absolute error", absolute_error / test_size)
    print("Mean relative error", relative_error * 100 / (test_size * batch))

    plt.plot(relative_error_list)
    plt.xlabel('batch')
    plt.ylabel('relative error')
    plt.savefig("picture/" + filename + "_test1.png")
    plt.clf()

    plt.plot(absolute_error_list)
    plt.xlabel('batch')
    plt.ylabel('absolute error')
    plt.savefig("picture/" + filename + "_test2.png")
