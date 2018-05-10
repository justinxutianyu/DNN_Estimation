# -*- coding: utf-8 -*-
# @Author: JustinXu
# @Date:   2018-05-09 20:11:41
# @Last Modified by:   JustinXu
# @Last Modified time: 2018-05-09 20:42:53

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

def layer_test(city, layers):
	optimize_time = time.strftime("%Y%m%d-%H%M%S")

	layer_labels = [str(layers[i]) + "layers" for i in range(4)]
	loss_array_list = []
	rel_error_list = []
	abs_error_list = []
	for i in range(4):
		adam_loss_array, adam_rel_error, adam_abs_error = compute.dif_layer_compute(city, "Adam", layers[i])
		loss_array_list.append(adam_loss_array)
		rel_error_list.append(adam_rel_error)
		abs_error_list.append(adam_abs_error)


	for i in range(4):
		plt.plot(loss_array_list[i])
	plt.legend(layer_labels, loc='upper right')
	plt.savefig("picture/"+city.location+"_"+optimize_time+"_layers_loss_test.png")
	plt.clf()

	for i in range(4):
		plt.plot(rel_error_list[i])
	plt.legend(layer_labels, loc='upper right')
	plt.savefig("picture/"+city.location+"_"+optimize_time+"_layers_relerror_test.png")
	plt.clf()

	for i in range(4):
		plt.plot(abs_error_list[i])
	plt.legend(layer_labels, loc='upper right')
	plt.savefig("picture/"+city.location+"_"+optimize_time+"_layers_abserror_test.png")
	plt.clf()


layers = [3,4,5,6]
city = city.City('Mel')
layer_test(city, layers)

city = city.City('NY')
layer_test(city, layers)
