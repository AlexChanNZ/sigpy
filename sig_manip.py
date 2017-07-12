# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
from __future__ import division # Correct Python 2 division

import numpy as np
import scipy.io as sio

import config_global as cg




def normalise_chan_data(chanData):

	print("chanData.shape",chanData.shape)
	normedData = (chanData - chanData.mean(axis=0)) / chanData.std(axis=0)
	print("normedData.shape",normedData.shape)

	return normedData




def preprocess(chanData):

	return normalise_chan_data(chanData)


