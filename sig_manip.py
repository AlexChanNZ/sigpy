# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
from __future__ import division # Correct Python 2 division

import numpy as np
import scipy.io as sio

import config_global as cg
from sklearn.preprocessing import scale # for normalisation between 0 and 1



def norm_first_clip(inData, minVal, maxVal) :
	outData = np.clip(inData, minVal, maxVal)
	print("First clip: ",outData[0,200:240])
	print("outData.shape: ", outData.shape)

	return outData



def norm_second_clip(secData) : #Shameer's secret sauce
	massagedData = (secData - secData.mean(axis=0)) * (4*(secData.std(axis=0)))
	print("massagedData: ", massagedData[200:240])
	print("massagedData.shape: ", massagedData.shape)

	return massagedData



def normalise_chan_data(chanData) :
	# Clip between -2, 2 mv and Shameers normalisation
	clippedData = norm_second_clip(norm_first_clip(chanData,-2000, 2000))

	# Classic normalisation [0,1]
	normedData = (clippedData - np.max(clippedData))/-np.ptp(clippedData)
	print("Normed data:" , normedData[200:240])
	print("normedData.shape: ", normedData.shape)

	return normedData


def preprocess(chanData) :
	return normalise_chan_data(chanData)


