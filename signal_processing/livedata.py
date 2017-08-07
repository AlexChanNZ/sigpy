# -*- coding: utf-8 -*-
"""
Author : Terry 
"""
from __future__ import division # Correct Python 2 division

import numpy as np

import config_global as cg


from threading import Thread
from multiprocessing.dummy import Pool as ThreadPool 

import time



class LiveData(Thread):

	def __init__(self, bufferSizeInSamples=30, dSource=None, dSourceSampleRate=None):
		Thread.__init__(self)
		self.startTime = time.time()
		self.bufferSizeInSamples = bufferSizeInSamples

		if dSourceSampleRate:
			self.dSourceSampleRate = dSourceSampleRate
		else:
			self.dSourceSampleRate = 1 / cg.dat['SigPy']['timeBetweenSamples']

		self.stampToIndexMultiplier = 1 / self.dSourceSampleRate

		# set time to sleep for roughly as long as it takes to build up the buffer
		self.timeToSleep = self.bufferSizeInSamples / self.dSourceSampleRate 
		self.timeTosleep = 0.1

		#if a data source was specified
		if dSource: 
			self.dataSource = dSource

		#else use loaded data to simulate online
		else:
			self.dataSource = cg.dat['SigPy']['filtData'] 
		
		self.lastIndexPush = 0
		self.currIndex = 0
		self.bufferedData = np.array([])



	def run(self):
		''' Start pulling in live data (or simulation of live data) '''
		print("Starting thread for live data capture:")

		while True:
			print("self.currIndex: ", self.currIndex)
			# Get time since capture began
			self.currTime = time.time() - self.startTime
			self.currIndex = int(self.currTime * self.dSourceSampleRate)

			nSamplesBuffered = self.currIndex - self.lastIndexPush
			print("nSamplesBuffered: ", nSamplesBuffered)


			# If caught more samples than buffer requested, push data to animation
			if nSamplesBuffered >= self.bufferSizeInSamples:

				self.bufferedData = self.dataSource[:,self.lastIndexPush: self.currIndex]
				self.lastIndexPush = self.currIndex


			# sleep 
			time.sleep(self.timeToSleep)










