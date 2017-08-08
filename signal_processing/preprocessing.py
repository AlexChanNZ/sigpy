# -*- coding: utf-8 -*-
"""
Author : Shameer Sathar
"""
from __future__ import division # Correct Python 2 division

import numpy as np
import scipy.io as sio

import config_global as cg

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn



def norm_first_clip(inData, minVal, maxVal) :

	outData = np.clip(inData, minVal, maxVal)

	return outData



def norm_second_clip(secData) : #Shameer's secret sauce

	massagedData = (secData - secData.mean(axis=0)) * (4*(secData.std(axis=0)))
	# print("massagedData.shape: ", massagedData.shape)

	return massagedData



def normalise_chan_data(chanData) :
	# Clip between -2, 2 mv and perform Shameers normalisation
	clippedData = norm_second_clip(norm_first_clip(chanData,-2000, 2000))

	# Classic normalisation [0,1]

	normaliseData = np.apply_along_axis(sklearn.preprocessing.MinMaxScaler().fit_transform, 1, clippedData)

	return normaliseData



def preprocess(chanData) :
	return normalise_chan_data(chanData)



def indices_to_timestamps(indices):
	return indices * timeBetweenSamples



