# -*- coding: utf-8 -*-
# @Author: Steven_Xu
# @Date:   2018-05-11 11:53:11
# @Last Modified by:   Steven_Xu
# @Last Modified time: 2018-05-29 14:40:05
# This file is to compare our model with DiDi's method
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


def layer(x, input_size, output_size, activation):
    W = tf.Variable(tf.truncated_normal(
        [input_size, output_size], mean=0.0, stddev=0.01))
    b = tf.Variable(tf.truncated_normal([output_size]))
    if activation == 'relu':
        h = tf.nn.relu(tf.matmul(x, W) + b)
    if activation == 'sigmoid':
        h = tf.nn.sigmoid(tf.matmul(x, W) + b)
    return h


def compute(city, optimizer_flag, flag):
    ######################## set parameters ##################
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Intialize class city
    # city = city.City('Mel')

    filename = optimizer_flag + "_" + city.name(timestr)

    ########################  loading data and graph #######################
    path = "/mnt/Project/data"
    distance_matrix, test_distance_matrix, max_distance = util.load_data(
        city, path)

    ######################## set some variables #######################
    input_size = 2 * city.d
    unit = city.unit
    output_size = 1
    # inpute features
    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y_ = tf.placeholder(
        tf.float32, [None, output_size], name='y')  # predictions

    # hidden layer 1
    # W1 = tf.Variable(tf.truncated_normal(
    #     [2 * city.d, city.unit], mean=0.0, stddev=0.01), name='W1')
    # b1 = tf.Variable(tf.truncated_normal([city.unit]), name='b1')
    if flag == 'didi':
        h1 = layer(x, input_size, 20, 'relu')
        h2 = layer(h1, 20, 100, 'relu')
        y = layer(h2, 100, output_size, 'sigmoid')
    if flag == 'our':
        h1 = layer(x, input_size, unit, 'relu')
        h2 = layer(h1, unit, unit, 'relu')
        h3 = layer(h2, unit, unit, 'relu')
        y = layer(h3, unit, output_size, 'sigmoid')

    ####################### Loss Function  #########################
    mse = tf.losses.mean_squared_error(y, y_)
    # error = tf.reduce_mean(tf.abs(tf.subtract(y,y_)))

    # sys.exit(0)
    if optimizer_flag == "GradientDescent":
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=city.learning_rate).minimize(mse)

    if optimizer_flag == "Adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate=city.learning_rate).minimize(mse)

    if optimizer_flag == "RMSProp":
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=city.learning_rate).minimize(mse)

    if optimizer_flag == "Momentum":
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=city.learning_rate, momentum=0.9).minimize(mse)

    # ####################### Saver         #########################
    saver = tf.train.Saver()

    # ###################### Initialize, Accuracy and Run #################
    print('Start training')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, "data/" + filename + "_model" + ".ckpt")
        loss_array = []
        size = city.size
        d = city.d
        epoch = city.epoch
        batch = city.batch_size
        for k in range(epoch):
            index = util.shuffle_batch(size, batch)
            for i in range(batch):
                # batch_xs, batch_ys = # mnist.train.next_batch(100)
                # print(str(i+1) + "th training")
                batch_xs = np.zeros(shape=(size, 2 * d))
                batch_ys = np.zeros(shape=(size, 1))
                for j in range(size):
                    vi = np.squeeze(np.asarray(
                        distance_matrix[index[i * size + j, 0], :]))
                    vj = np.squeeze(np.asarray(
                        distance_matrix[index[i * size + j, 1], :]))

                    batch_xs[j] = np.concatenate([vi, vj])
                    batch_ys[j] = test_distance_matrix[
                        index[i * size + j, 0], index[i * size + j, 1]]

                _, c = sess.run([optimizer, mse], feed_dict={
                                x: batch_xs, y_: batch_ys})
                loss_array.append(c)
                print('train_step:', (i + 1), 'cost =', '{:.6f}'.format(c))

        plt.plot(loss_array)
        plt.ylabel('nn_loss')
        plt.savefig("picture/" + filename + '_loss.png')
        plt.clf()

        # Load testing data
        test_size = int(size / 10)
        d = city.d
        # temporal variable
        dif = []
        cost = 0
        absolute_error = 0.0
        relative_error = 0.0
        true_distance = 0.0
        pred_distance = 0.0
        relative_error_list = []
        absolute_error_list = []

        index = np.asarray(range(size))
        np.random.shuffle(index)
        index = index[:test_size]
        for k in range(test_size):
            i = index[k]
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

            print('test_step:', (k + 1),
                  'mean squared error =', '{:.6f}'.format(c))
            batch_relative_error = batch_relative_error / batch * 100
            relative_error_list.append(batch_relative_error)
            print('test_step:', (k + 1), 'relative error =',
                  '{:.6f}'.format(batch_relative_error))
            e = mean_absolute_error(pred_y, actual_y)
            absolute_error += e
            absolute_error_list.append(e)
            print('test_step:', (k + 1),
                  'absolute error =', '{:.6f}'.format(e))

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
        plt.clf()

        return (loss_array, relative_error_list, absolute_error_list)
