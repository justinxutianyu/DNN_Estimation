# -*- coding: utf-8 -*-
# @Author: JustinXu
# @Date:   2018-05-08 22:18:56
# @Last Modified by:   xutianyu
# @Last Modified time: 2018-05-11 00:46:06

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
import compute


def input_test(city):
    optimize_time = time.strftime("%Y%m%d-%H%M%S")

    labels = ['Adjacency Matrix', 'Distance Matrix', 'Landmark Distance']
    loss_array_list = []
    rel_error_list = []
    abs_error_list = []

    flag = 'adj'
    adj_loss_array, adj_rel_error, adj_abs_error = compute.dif_input_compute(
        city, "Adam", flag)
    loss_array_list.append(adj_loss_array)
    rel_error_list.append(adj_rel_error)
    abs_error_list.append(adj_abs_error)

    flag = 'all'
    all_loss_array, all_rel_error, all_abs_error = compute.dif_input_compute(
        city, "Adam", flag)
    loss_array_list.append(all_loss_array)
    rel_error_list.append(all_rel_error)
    abs_error_list.append(all_abs_error)

    flag = 'landmark'
    landmark_loss_array, landmark_rel_error, landmark_abs_error = compute.dif_input_compute(
        city, "Adam", flag)
    loss_array_list.append(landmark_loss_array)
    rel_error_list.append(landmark_rel_error)
    abs_error_list.append(landmark_abs_error)

    for i in range(3):
        plt.plot(loss_array_list[i])
    plt.legend(labels, loc='upper right')
    plt.savefig("picture/" + city.location + "_" +
                optimize_time + "_input_loss_test.png")
    plt.clf()

    for i in range(3):
        plt.plot(rel_error_list[i])
    plt.legend(labels, loc='upper right')
    plt.savefig("picture/" + city.location + "_" +
                optimize_time + "_input_relerror_test.png")
    plt.clf()

    for i in range(3):
        plt.plot(abs_error_list[i])
    plt.legend(labels, loc='upper right')
    plt.savefig("picture/" + city.location + "_" +
                optimize_time + "_input_abserror_test.png")
    plt.clf()

# city = city.City('Mel')
# input_test(city)

city = city.City('NY')
input_test(city)
