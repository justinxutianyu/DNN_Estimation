# -*- coding: utf-8 -*-
# @Author: tiany
# @Date:   2018-05-07 13:42:23
# @Last Modified by:   JustinXu
# @Last Modified time: 2018-05-08 12:58:29

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

optimize_time = time.strftime("%Y%m%d-%H%M%S")
city = city.City('Mel')

landmark_list = [100, 300,500, 700, 900, 2000]


mom_loss_array, mom_rel_error, mom_abs_error = compute.compute(city, "Momentum")

gd_loss_array, gd_rel_error, gd_abs_error = compute.compute(city, "GradientDescent")

adam_loss_array, adam_rel_error, adam_abs_error = compute.compute(city, "Adam")

rms_loss_array, rms_rel_error, rms_abs_error = compute.compute(city, "RMSProp")



plt.plot(gd_loss_array)
plt.plot(adam_loss_array)
plt.plot(rms_loss_array)
plt.plot(mom_loss_array)

plt.legend(['GradientDescent', 'Adam', 'RMSProp', 'Momentum'], loc='upper right')
plt.savefig("picture/"+optimize_time+"_optimizer_loss_test.png")
plt.clf()

plt.plot(gd_rel_error)
plt.plot(adam_rel_error)
plt.plot(rms_rel_error)
plt.plot(mom_rel_error)

plt.legend(['GradientDescent', 'Adam', 'RMSProp', 'Momentum'], loc='upper right')
plt.savefig("picture/"+optimize_time+"_optimizer_rel_test.png")
plt.clf()

plt.plot(gd_abs_error)
plt.plot(adam_abs_error)
plt.plot(rms_abs_error)
plt.plot(mom_abs_error)

plt.legend(['GradientDescent', 'Adam', 'RMSProp', 'Momentum'], loc='upper right')
plt.savefig("picture/"+optimize_time+"_optimizer_abs_test.png")
plt.clf()
