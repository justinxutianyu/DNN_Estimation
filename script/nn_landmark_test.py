# -*- coding: utf-8 -*-
# @Author: JustinXu
# @Date:   2018-05-08 12:31:28
# @Last Modified by:   JustinXu
# @Last Modified time: 2018-05-08 22:23:11

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

def landmark_test(city, landmark_list):
	optimize_time = time.strftime("%Y%m%d-%H%M%S")

	landmark_label = ['landmark'+str(landmark_list[i]) for i in range(6)]
	loss_array_list = []
	rel_error_list = []
	abs_error_list = []
	for i in range(6):
		city.d = landmark_list[i]
		adam_loss_array, adam_rel_error, adam_abs_error = compute.compute(city, "Adam")
		loss_array_list.append(adam_loss_array)
		rel_error_list.append(adam_rel_error)
		abs_error_list.append(adam_abs_error)


	for i in range(6):
		plt.plot(loss_array_list[i])
	plt.legend(landmark_label, loc='upper right')
	plt.savefig("picture/"+city.location+"_"+optimize_time+"_landmark_loss_test.png")
	plt.clf()

	for i in range(6):
		plt.plot(rel_error_list[i])
	plt.legend(landmark_label, loc='upper right')
	plt.savefig("picture/"+city.location+"_"+optimize_time+"_landmark_relerror_test.png")
	plt.clf()

	for i in range(6):
		plt.plot(abs_error_list[i])
	plt.legend(landmark_label, loc='upper right')
	plt.savefig("picture/"+city.location+"_"+optimize_time+"_landmark_abserror_test.png")
	plt.clf()

# city = city.City('Mel')
# landmark_list = [100, 300, 500, 700, 900, 2000]
# landmark_test(city, landmark_list)

city = city.City('NY')
landmark_list = [100, 500, 1000, 1500, 2000, 2500]
landmark_test(city, landmark_list)