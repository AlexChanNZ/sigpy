# -*- coding: utf-8 -*-
"""
Author : Terry 
"""
from __future__ import division # Correct Python 2 division

import numpy as np

import config_global as cg


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
			self.dataSource = cg.dat['SigPy']['filtData'] 			
			self.dSourceSampleRate = 1 / cg.dat['SigPy']['timeBetweenSamples']
			self.timeBetweenSamples = cg.dat['SigPy']['timeBetweenSamples']
			self.maxIndex = self.dataSource.shape[1]
			self.bufferedChunk = self.dataSource[:,0]



		self.stampToIndexMultiplier = 1 / self.dSourceSampleRate

		# set time to sleep for roughly as long as it takes to build up the buffer
		# self.timeToSleep = self.bufferChunkSize / self.dSourceSampleRate 
		self.timeToSleep = 0.01
		
		self.lastIndex = 0
		self.bufferedChunk = np.zeros(shape=(nChans,1))
		self.newChunk = False

		self.lastCaptureTime = time.time()


	def run(self):
		''' Start pulling in live data (or simulation of live data) '''
		print("Starting thread for live data capture:")
		self.lastCaptureTime = time.time()

		while True:

			# Get time since capture began
			if (time.time() - self.lastCaptureTime) >= self.timeBetweenSamples :

				self.lastIndex += 1 				
				# print("self.bufferedChunk.shape: ", self.bufferedChunk.shape, "self.dataSource[:,self.lastIndex].shape: ",self.dataSource[:,self.lastIndex].shape)
				# self.bufferedChunk = np.append([self.bufferedChunk], [self.dataSource[:,self.lastIndex]], axis=1)
				if self.bufferedChunk.shape[0] > 1 :
					self.bufferedChunk = np.hstack((self.bufferedChunk, self.dataSource[:,self.lastIndex].reshape(-1,1))) #, axis=1)

				else:
					self.bufferedChunk = self.dataSource[:,self.lastIndex].reshape(-1,1)
				time.sleep(self.timeToSleep) # Thinking behind this is: 
				# The buffer is full, give the other thread some time to do the pre-processing
				self.lastCaptureTime = self.lastCaptureTime + self.timeBetweenSamples
				

				# print("Self.lastIndex: ", self.lastIndex)
			# else:
			# 	time.sleep(self.timeToSleep)



			if self.lastIndex >= self.maxIndex :
				return

			# nSamplesBuffered = self.currIndex - self.lastIndexPush
			# print("nSamplesBuffered: ", nSamplesBuffered)

			# # If caught more samples than buffer requested, push data to animation
			# if nSamplesBuffered >= self.bufferChunkSize:
			# 	self.bufferedChunk = self.dataSource[:,self.lastIndexPush: self.currIndex]
			# 	self.lastIndexPush = self.currIndex

			# sleep 










