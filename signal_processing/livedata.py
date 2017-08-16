# -*- coding: utf-8 -*-
"""
Author : Terry 
"""
from __future__ import division # Correct Python 2 division

import numpy as np

import config_global as sp


from threading import Thread

import time


class LiveData(Thread):

	def __init__(self, nChans=224, dSource=None, dSourceSampleRate=None):
		Thread.__init__(self)
		self.startTime = time.time()
		# self.bufferChunkSize = bufferChunkSize

		if dSource:
			self.dataSource = dSource			
			self.timeBetweenSamples = 1 / self.dSourceSampleRate
			self.maxIndex = 999999999999

		else:
			self.dataSource = sp.dat['SigPy']['dataFilt'] 			
			self.dSourceSampleRate = 1 / sp.dat['SigPy']['timeBetweenSamples']
			self.timeBetweenSamples = sp.dat['SigPy']['timeBetweenSamples']
			self.maxIndex = self.dataSource.shape[1]
			self.bufferedChunk = self.dataSource[:,0]


		self.nChans = self.dataSource.shape[0]


		self.stampToIndexMultiplier = 1 / self.dSourceSampleRate
		self.priorBufferedChunk = np.copy(self.bufferedChunk).reshape(-1,1)

		# set time to sleep for roughly as long as it takes to build up the buffer
		# self.timeToSleep = self.bufferChunkSize / self.dSourceSampleRate 
		self.timeToSleep = 0.1
		
		self.lastIndex = 0
		self.bufferedChunk = np.zeros(shape=(self.nChans,1))

		self.newChunk = False
		self.shouldStop = False

		self.lastCaptureTime = time.time()


	def run(self):
		''' Start pulling in live data (or simulation of live data) '''
		print("Starting thread for live data capture:")
		self.lastCaptureTime = time.time()

		while True:

			# Get time since capture began
			if (time.time() - self.lastCaptureTime) >= self.timeBetweenSamples :

				self.lastIndex += 1 				

				#Add frame to buffer

				self.bufferedChunk = np.hstack((self.bufferedChunk, self.dataSource[:,self.lastIndex].reshape(-1,1))) #, axis=1)

				# Update capture time
				self.lastCaptureTime = self.lastCaptureTime + self.timeBetweenSamples

			if self.lastIndex >= self.maxIndex :
				print("Live capture thread terminating, reached last index.")
				return

			if self.shouldStop :
				print("Live capture thread terminating, signaled to stop by another thread.")
				return

			# nSamplesBuffered = self.currIndex - self.lastIndexPush
			# print("lastIndex: ", self.lastIndex)











