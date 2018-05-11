# -*- coding: utf-8 -*-
# @Author: tiany
# @Date:   2018-05-07 13:42:23
# @Last Modified time: 2018-05-11 11:56:13

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
import benchmark_compute

optimize_time = time.strftime("%Y%m%d-%H%M%S")

newyork = city.City('NY')

didi_loss_array, didi_rel_error, didi_abs_error = benchmark_compute.compute(
    newyork, "Adam", "didi")

our_loss_array, our_rel_error, our_abs_error = benchmark_compute.compute(
    newyork, "Adam", "our")

plt.plot(didi_loss_array)
plt.plot(our_loss_array)

plt.legend(['DiDi', 'Our Method'], loc='upper right')
plt.savefig("picture/" + optimize_time + "_benchmark_rel_test.png")
plt.clf()

plt.plot(didi_rel_error)
plt.plot(our_rel_error)

plt.legend(['DiDi', 'Our Method'], loc='upper right')
plt.savefig("picture/" + optimize_time + "_benchmark_rel_test.png")
plt.clf()

plt.plot(didi_abs_error)
plt.plot(our_abs_error)

plt.legend(['DiDi', 'Our Method'], loc='upper right')
plt.savefig("picture/" + optimize_time + "_benchmark_abs_test.png")
plt.clf()

mel = city.City('Mel')

didi_loss_array, didi_rel_error, didi_abs_error = benchmark_compute.compute(
    newyork, "Adam", "didi")

our_loss_array, our_rel_error, our_abs_error = benchmark_compute.compute(
    newyork, "Adam", "our")

plt.plot(didi_loss_array)
plt.plot(our_loss_array)

plt.legend(['DiDi', 'Our Method'], loc='upper right')
plt.savefig("picture/" + optimize_time + "_benchmark_rel_test.png")
plt.clf()

plt.plot(didi_rel_error)
plt.plot(our_rel_error)

plt.legend(['DiDi', 'Our Method'], loc='upper right')
plt.savefig("picture/" + optimize_time + "_benchmark_rel_test.png")
plt.clf()

plt.plot(didi_abs_error)
plt.plot(our_abs_error)

plt.legend(['DiDi', 'Our Method'], loc='upper right')
plt.savefig("picture/" + optimize_time + "_benchmark_abs_test.png")
plt.clf()
