# -*- coding: utf-8 -*-
# @Author: Steven_Xu
# @Date:   2018-05-11 11:23:57
# @Last Modified by:   Steven_Xu
# @Last Modified time: 2018-05-11 11:24:10
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
